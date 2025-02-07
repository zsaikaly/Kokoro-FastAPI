"""Clean Kokoro implementation with controlled resource management."""

import os
from typing import AsyncGenerator, Optional, Union, Tuple

import numpy as np
import torch
from loguru import logger

from ..core import paths
from ..core.model_config import model_config
from ..core.config import settings
from .base import BaseModelBackend
from kokoro import KModel, KPipeline

class KokoroV1(BaseModelBackend):
    """Kokoro backend with controlled resource management."""

    def __init__(self):
        """Initialize backend with environment-based configuration."""
        super().__init__()
        # Strictly respect settings.use_gpu
        self._device = "cuda" if settings.use_gpu else "cpu"
        self._model: Optional[KModel] = None
        self._pipeline: Optional[KPipeline] = None

    async def load_model(self, path: str) -> None:
        """Load pre-baked model.
        
        Args:
            path: Path to model file
            
        Raises:
            RuntimeError: If model loading fails
        """
        try:
            # Get verified model path
            model_path = await paths.get_model_path(path)
            config_path = os.path.join(os.path.dirname(model_path), 'config.json')
            
            if not os.path.exists(config_path):
                raise RuntimeError(f"Config file not found: {config_path}")
            
            logger.info(f"Loading Kokoro model on {self._device}")
            logger.info(f"Config path: {config_path}")
            logger.info(f"Model path: {model_path}")
            
            # Load model and let KModel handle device mapping
            self._model = KModel(
                config=config_path,
                model=model_path
            ).eval()
            # Move to CUDA if needed
            if self._device == "cuda":
                self._model = self._model.cuda()
            
            # Initialize pipeline with our model and device
            self._pipeline = KPipeline(
                lang_code='a',
                model=self._model,  # Pass our model directly
                device=self._device  # Match our device setting
            )
        except FileNotFoundError as e:
            raise e
        except Exception as e:
            raise RuntimeError(f"Failed to load Kokoro model: {e}")

    async def generate_from_tokens(
        self,
        tokens: str,
        voice: Union[str, Tuple[str, Union[torch.Tensor, str]]],
        speed: float = 1.0
    ) -> AsyncGenerator[np.ndarray, None]:
        """Generate audio from phoneme tokens.

        Args:
            tokens: Input phoneme tokens to synthesize
            voice: Either a voice path string or a tuple of (voice_name, voice_tensor/path)
            speed: Speed multiplier

        Yields:
            Generated audio chunks

        Raises:
            RuntimeError: If generation fails
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")

        try:
            # Memory management for GPU
            if self._device == "cuda":
                if self._check_memory():
                    self._clear_memory()

            # Handle voice input
            voice_path: str
            if isinstance(voice, tuple):
                voice_name, voice_data = voice
                if isinstance(voice_data, str):
                    voice_path = voice_data
                else:
                    # Save tensor to temporary file
                    import tempfile
                    temp_dir = tempfile.gettempdir()
                    voice_path = os.path.join(temp_dir, f"{voice_name}.pt")
                    # Save tensor with CPU mapping for portability
                    torch.save(voice_data.cpu(), voice_path)
            else:
                voice_path = voice

            # Load voice tensor with proper device mapping
            voice_tensor = await paths.load_voice_tensor(voice_path, device=self._device)
            # Save back to a temporary file with proper device mapping
            import tempfile
            temp_dir = tempfile.gettempdir()
            temp_path = os.path.join(temp_dir, f"temp_voice_{os.path.basename(voice_path)}")
            await paths.save_voice_tensor(voice_tensor, temp_path)
            voice_path = temp_path

            # Generate using pipeline's generate_from_tokens method
            logger.debug(f"Generating audio from tokens: '{tokens[:100]}...'")
            for result in self._pipeline.generate_from_tokens(
                tokens=tokens,
                voice=voice_path,
                speed=speed,
                model=self._model
            ):
                if result.audio is not None:
                    logger.debug(f"Got audio chunk with shape: {result.audio.shape}")
                    yield result.audio.numpy()
                else:
                    logger.warning("No audio in chunk")

        except Exception as e:
            logger.error(f"Generation failed: {e}")
            if (
                self._device == "cuda"
                and model_config.pytorch_gpu.retry_on_oom
                and "out of memory" in str(e).lower()
            ):
                self._clear_memory()
                async for chunk in self.generate_from_tokens(tokens, voice, speed):
                    yield chunk
            raise

    async def generate(
        self,
        text: str,
        voice: Union[str, Tuple[str, Union[torch.Tensor, str]]],
        speed: float = 1.0
    ) -> AsyncGenerator[np.ndarray, None]:
        """Generate audio using model.

        Args:
            text: Input text to synthesize
            voice: Either a voice path string or a tuple of (voice_name, voice_tensor/path)
            speed: Speed multiplier

        Yields:
            Generated audio chunks

        Raises:
            RuntimeError: If generation fails
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")

        try:
            # Memory management for GPU
            if self._device == "cuda":
                if self._check_memory():
                    self._clear_memory()

            # Handle voice input
            voice_path: str
            if isinstance(voice, tuple):
                voice_name, voice_data = voice
                if isinstance(voice_data, str):
                    voice_path = voice_data
                else:
                    # Save tensor to temporary file
                    import tempfile
                    temp_dir = tempfile.gettempdir()
                    voice_path = os.path.join(temp_dir, f"{voice_name}.pt")
                    # Save tensor with CPU mapping for portability
                    torch.save(voice_data.cpu(), voice_path)
            else:
                voice_path = voice

            # Load voice tensor with proper device mapping
            voice_tensor = await paths.load_voice_tensor(voice_path, device=self._device)
            # Save back to a temporary file with proper device mapping
            import tempfile
            temp_dir = tempfile.gettempdir()
            temp_path = os.path.join(temp_dir, f"temp_voice_{os.path.basename(voice_path)}")
            await paths.save_voice_tensor(voice_tensor, temp_path)
            voice_path = temp_path

            # Generate using pipeline, force model to prevent downloads
            logger.debug(f"Generating audio for text: '{text[:100]}...'")
            for result in self._pipeline(text, voice=voice_path, speed=speed, model=self._model):
                if result.audio is not None:
                    logger.debug(f"Got audio chunk with shape: {result.audio.shape}")
                    yield result.audio.numpy()
                else:
                    logger.warning("No audio in chunk")

        except Exception as e:
            logger.error(f"Generation failed: {e}")
            if (
                self._device == "cuda"
                and model_config.pytorch_gpu.retry_on_oom
                and "out of memory" in str(e).lower()
            ):
                self._clear_memory()
                async for chunk in self.generate(text, voice, speed):
                    yield chunk
            raise

    def _check_memory(self) -> bool:
        """Check if memory usage is above threshold."""
        if self._device == "cuda":
            memory_gb = torch.cuda.memory_allocated() / 1e9
            return memory_gb > model_config.pytorch_gpu.memory_threshold
        return False

    def _clear_memory(self) -> None:
        """Clear device memory."""
        if self._device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    def unload(self) -> None:
        """Unload model and free resources."""
        if self._model is not None:
            del self._model
            self._model = None
        if self._pipeline is not None:
            del self._pipeline
            self._pipeline = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._model is not None and self._pipeline is not None

    @property
    def device(self) -> str:
        """Get device model is running on."""
        return self._device
