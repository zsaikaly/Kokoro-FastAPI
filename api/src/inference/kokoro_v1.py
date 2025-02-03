"""PyTorch inference backend with environment-based configuration."""

import gc
import os
from typing import AsyncGenerator, Optional, List, Union, Tuple
from contextlib import nullcontext

import numpy as np
import torch
from loguru import logger

from ..core import paths
from ..core.model_config import model_config
from ..core.config import settings
from .base import BaseModelBackend
from kokoro import KModel, KPipeline


class KokoroV1(BaseModelBackend):
    """Kokoro package based inference backend with environment-based configuration."""

    def __init__(self):
        """Initialize backend based on environment configuration."""
        super().__init__()

        # Configure device based on settings
        self._device = (
            "cuda" if settings.use_gpu and torch.cuda.is_available() else "cpu"
        )
        self._model: Optional[KModel] = None
        self._pipeline: Optional[KPipeline] = None

    async def load_model(self, path: str) -> None:
        """Load Kokoro model.
        
        Args:
            path: Path to model file
            
        Raises:
            RuntimeError: If model loading fails
        """
        try:
            # Get verified model path
            model_path = await paths.get_model_path(path)
            
            # Get config.json path from the same directory
            config_path = os.path.join(os.path.dirname(model_path), 'config.json')
            
            if not os.path.exists(config_path):
                raise RuntimeError(f"Config file not found: {config_path}")
            
            logger.info(f"Loading Kokoro model on {self._device}")
            logger.info(f"Config path: {config_path}")
            logger.info(f"Model path: {model_path}")
            
            # Initialize model with config and weights
            self._model = KModel(config=config_path, model=model_path).to(self._device).eval()
            # Initialize pipeline with American English by default
            self._pipeline = KPipeline(lang_code='a', model=self._model, device=self._device)
            
        except Exception as e:
            raise RuntimeError(f"Failed to load Kokoro model: {e}")

    async def generate(
        self, text: str, voice: Union[str, Tuple[str, Union[torch.Tensor, str]]], speed: float = 1.0
    ) -> AsyncGenerator[np.ndarray, None]:
        """Generate audio using model.

        Args:
            text: Input text to synthesize
            voice: Either a voice path string or a tuple of (voice_name, voice_tensor_or_path)
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
            if isinstance(voice, str):
                voice_path = voice  # Voice path provided directly
                logger.debug(f"Using voice path directly: {voice_path}")
                # Get language code from first letter of voice name
                try:
                    name = os.path.basename(voice_path)
                    logger.debug(f"Voice basename: {name}")
                    if name.endswith('.pt'):
                        name = name[:-3]
                    lang_code = name[0]
                    logger.debug(f"Extracted language code: {lang_code}")
                except Exception as e:
                    # Default to American English if we can't get language code
                    logger.warning(f"Failed to extract language code: {e}, defaulting to 'a'")
                    lang_code = 'a'
            else:
                # Unpack voice name and tensor/path
                voice_name, voice_data = voice
                # If voice_data is a path, use it directly
                if isinstance(voice_data, str):
                    voice_path = voice_data
                    logger.debug(f"Using provided voice path: {voice_path}")
                else:
                    # Save tensor to temporary file
                    import tempfile
                    temp_dir = tempfile.gettempdir()
                    voice_path = os.path.join(temp_dir, f"{voice_name}.pt")
                    logger.debug(f"Saving voice tensor to: {voice_path}")
                    torch.save(voice_data, voice_path)
                # Get language code from voice name
                lang_code = voice_name[0]
                logger.debug(f"Using language code '{lang_code}' from voice name {voice_name}")

            # Update pipeline's language code if needed
            if self._pipeline.lang_code != lang_code:
                logger.debug(f"Creating pipeline with lang_code='{lang_code}'")
                self._pipeline = KPipeline(lang_code=lang_code, model=self._model, device=self._device)

            # Generate audio using pipeline
            logger.debug(f"Generating audio for text: '{text[:100]}...'")
            for i, result in enumerate(self._pipeline(text, voice=voice_path, speed=speed)):
                logger.debug(f"Processing chunk {i+1}")
                if result.audio is not None:
                    logger.debug(f"Got audio chunk {i+1} with shape: {result.audio.shape}")
                    yield result.audio.numpy()
                else:
                    logger.warning(f"No audio in chunk {i+1}")

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
        finally:
            if self._device == "cuda" and model_config.pytorch_gpu.sync_cuda:
                torch.cuda.synchronize()

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
        gc.collect()

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
        gc.collect()

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._model is not None and self._pipeline is not None

    @property
    def device(self) -> str:
        """Get device model is running on."""
        return self._device
