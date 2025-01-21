"""Model management and caching."""

import os
from typing import Dict, List, Optional, Union

import torch
from loguru import logger
from pydantic import BaseModel

from .base import BaseModelBackend
from .voice_manager import get_manager as get_voice_manager
from .onnx_cpu import ONNXCPUBackend
from .onnx_gpu import ONNXGPUBackend
from .pytorch_cpu import PyTorchCPUBackend
from .pytorch_gpu import PyTorchGPUBackend
from ..core import paths
from ..core.config import settings
from ..structures.model_schemas import ModelConfig


class ModelManager:
    """Manages model loading and inference across backends."""

    def __init__(self, config: Optional[ModelConfig] = None):
        """Initialize model manager.
        
        Args:
            config: Optional configuration
        """
        self._config = config or ModelConfig()
        self._backends: Dict[str, BaseModelBackend] = {}
        self._current_backend: Optional[str] = None
        self._voice_manager = get_voice_manager()
        self._initialize_backends()

    def _initialize_backends(self) -> None:
        """Initialize available backends."""
        """Initialize available backends."""
        # Initialize GPU backends if available
        if settings.use_gpu and torch.cuda.is_available():
            try:
                # PyTorch GPU
                self._backends['pytorch_gpu'] = PyTorchGPUBackend()
                self._current_backend = 'pytorch_gpu'
                logger.info("Initialized PyTorch GPU backend")
                
                # ONNX GPU
                self._backends['onnx_gpu'] = ONNXGPUBackend()
                logger.info("Initialized ONNX GPU backend")
            except Exception as e:
                logger.error(f"Failed to initialize GPU backends: {e}")
                # Fallback to CPU if GPU fails
                self._initialize_cpu_backends()
        else:
            self._initialize_cpu_backends()

    def _initialize_cpu_backends(self) -> None:
        """Initialize CPU backends."""
        try:
            # PyTorch CPU
            self._backends['pytorch_cpu'] = PyTorchCPUBackend()
            self._current_backend = 'pytorch_cpu'
            logger.info("Initialized PyTorch CPU backend")
            
            # ONNX CPU
            self._backends['onnx_cpu'] = ONNXCPUBackend()
            logger.info("Initialized ONNX CPU backend")
        except Exception as e:
            logger.error(f"Failed to initialize CPU backends: {e}")
            raise RuntimeError("No backends available")

    def get_backend(self, backend_type: Optional[str] = None) -> BaseModelBackend:
        """Get specified backend.
        
        Args:
            backend_type: Backend type ('pytorch_cpu', 'pytorch_gpu', 'onnx_cpu', 'onnx_gpu'),
                         uses default if None
            
        Returns:
            Model backend instance
            
        Raises:
            ValueError: If backend type is invalid
            RuntimeError: If no backends are available
        """
        if not self._backends:
            raise RuntimeError("No backends available")

        if backend_type is None:
            backend_type = self._current_backend
        
        if backend_type not in self._backends:
            raise ValueError(
                f"Invalid backend type: {backend_type}. "
                f"Available backends: {', '.join(self._backends.keys())}"
            )

        return self._backends[backend_type]

    def _determine_backend(self, model_path: str) -> str:
        """Determine appropriate backend based on model file.
        
        Args:
            model_path: Path to model file
            
        Returns:
            Backend type to use
        """
        is_onnx = model_path.lower().endswith('.onnx')
        has_gpu = settings.use_gpu and torch.cuda.is_available()
        
        if is_onnx:
            return 'onnx_gpu' if has_gpu else 'onnx_cpu'
        else:
            return 'pytorch_gpu' if has_gpu else 'pytorch_cpu'

    async def load_model(
        self,
        model_path: str,
        backend_type: Optional[str] = None
    ) -> None:
        """Load model on specified backend.
        
        Args:
            model_path: Path to model file
            backend_type: Backend to load on, uses default if None
            
        Raises:
            RuntimeError: If model loading fails
        """
        try:
            # Get absolute model path
            abs_path = await paths.get_model_path(model_path)
            
            # Auto-determine backend if not specified
            if backend_type is None:
                backend_type = self._determine_backend(abs_path)
            
            backend = self.get_backend(backend_type)
            
            # Load model and run warmup
            await backend.load_model(abs_path)
            logger.info(f"Loaded model on {backend_type} backend")
            await self._warmup_inference(backend)
            
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")
            
    async def _warmup_inference(self, backend: BaseModelBackend) -> None:
        """Run warmup inference to initialize model."""
        try:
            # Import here to avoid circular imports
            from ..text_processing import process_text
            
            # Load default voice for warmup
            voice = await self._voice_manager.load_voice(settings.default_voice, device=backend.device)
            logger.info(f"Loaded voice {settings.default_voice} for warmup")
            
            # Use real text
            text = "Testing text to speech synthesis."
            logger.info(f"Running warmup inference with voice: af")
            
            # Process through pipeline
            sequences = process_text(text)
            if not sequences:
                raise ValueError("Text processing failed")
            
            # Run inference
            backend.generate(sequences[0], voice, speed=1.0)
            
        except Exception as e:
            logger.warning(f"Warmup inference failed: {e}")
            raise

    async def generate(
        self,
        tokens: list[int],
        voice_name: str,
        speed: float = 1.0,
        backend_type: Optional[str] = None
    ) -> torch.Tensor:
        """Generate audio using specified backend.
        
        Args:
            tokens: Input token IDs
            voice_name: Name of voice to use
            speed: Speed multiplier
            backend_type: Backend to use, uses default if None
            
        Returns:
            Generated audio tensor
            
        Raises:
            RuntimeError: If generation fails
        """
        backend = self.get_backend(backend_type)
        if not backend.is_loaded:
            raise RuntimeError("Model not loaded")

        try:
            # Load voice using voice manager
            voice = await self._voice_manager.load_voice(voice_name, device=backend.device)
            
            # Generate audio
            return backend.generate(tokens, voice, speed)
            
        except Exception as e:
            raise RuntimeError(f"Generation failed: {e}")

    def unload_all(self) -> None:
        """Unload models from all backends."""
        for backend in self._backends.values():
            backend.unload()
        logger.info("Unloaded all models")

    @property
    def available_backends(self) -> list[str]:
        """Get list of available backends.
        
        Returns:
            List of backend names
        """
        return list(self._backends.keys())

    @property
    def current_backend(self) -> str:
        """Get current default backend.
        
        Returns:
            Backend name
        """
        return self._current_backend


# Module-level instance
_manager: Optional[ModelManager] = None


def get_manager(config: Optional[ModelConfig] = None) -> ModelManager:
    """Get or create global model manager instance.
    
    Args:
        config: Optional model configuration
        
    Returns:
        ModelManager instance
    """
    global _manager
    if _manager is None:
        _manager = ModelManager(config)
    return _manager