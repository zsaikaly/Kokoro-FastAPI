"""Model management and caching."""

from typing import Dict, Optional

import torch
from loguru import logger

from ..core import paths
from ..core.config import settings
from ..core.model_config import ModelConfig, model_config
from .base import BaseModelBackend
from .onnx_cpu import ONNXCPUBackend
from .onnx_gpu import ONNXGPUBackend
from .pytorch_cpu import PyTorchCPUBackend
from .pytorch_gpu import PyTorchGPUBackend


class ModelManager:
    """Manages model loading and inference across backends."""

    def __init__(self, config: Optional[ModelConfig] = None):
        """Initialize model manager.
        
        Args:
            config: Optional configuration
        """
        self._config = config or model_config
        self._backends: Dict[str, BaseModelBackend] = {}
        self._current_backend: Optional[str] = None
        self._initialize_backends()

    def _initialize_backends(self) -> None:
        """Initialize available backends based on settings."""
        has_gpu = settings.use_gpu and torch.cuda.is_available()
        
        try:
            if has_gpu:
                if settings.use_onnx:
                    # ONNX GPU primary
                    self._backends['onnx_gpu'] = ONNXGPUBackend()
                    self._current_backend = 'onnx_gpu'
                    logger.info("Initialized ONNX GPU backend")
                    
                    # PyTorch GPU fallback
                    self._backends['pytorch_gpu'] = PyTorchGPUBackend()
                    logger.info("Initialized PyTorch GPU backend")
                else:
                    # PyTorch GPU primary
                    self._backends['pytorch_gpu'] = PyTorchGPUBackend()
                    self._current_backend = 'pytorch_gpu'
                    logger.info("Initialized PyTorch GPU backend")
                    
                    # ONNX GPU fallback
                    self._backends['onnx_gpu'] = ONNXGPUBackend()
                    logger.info("Initialized ONNX GPU backend")
            else:
                self._initialize_cpu_backends()
        except Exception as e:
            logger.error(f"Failed to initialize GPU backends: {e}")
            # Fallback to CPU if GPU fails
            self._initialize_cpu_backends()

    def _initialize_cpu_backends(self) -> None:
        """Initialize CPU backends based on settings."""
        try:
            if settings.use_onnx:
                # ONNX CPU primary
                self._backends['onnx_cpu'] = ONNXCPUBackend()
                self._current_backend = 'onnx_cpu'
                logger.info("Initialized ONNX CPU backend")
                
                # PyTorch CPU fallback
                self._backends['pytorch_cpu'] = PyTorchCPUBackend()
                logger.info("Initialized PyTorch CPU backend")
            else:
                # PyTorch CPU primary
                self._backends['pytorch_cpu'] = PyTorchCPUBackend()
                self._current_backend = 'pytorch_cpu'
                logger.info("Initialized PyTorch CPU backend")
                
                # ONNX CPU fallback
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
        """Determine appropriate backend based on model file and settings.
        
        Args:
            model_path: Path to model file
            
        Returns:
            Backend type to use
        """
        has_gpu = settings.use_gpu and torch.cuda.is_available()
        
        # If ONNX is preferred or model is ONNX format
        if settings.use_onnx or model_path.lower().endswith('.onnx'):
            return 'onnx_gpu' if has_gpu else 'onnx_cpu'
        else:
            return 'pytorch_gpu' if has_gpu else 'pytorch_cpu'

    async def load_model(
        self,
        model_path: str,
        warmup_voice: Optional[torch.Tensor] = None,
        backend_type: Optional[str] = None
    ) -> None:
        """Load model on specified backend.
        
        Args:
            model_path: Path to model file
            warmup_voice: Optional voice tensor for warmup, skips warmup if None
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
            
            # Load model
            await backend.load_model(abs_path)
            logger.info(f"Loaded model on {backend_type} backend")
            
            # Run warmup if voice provided
            if warmup_voice is not None:
                await self._warmup_inference(backend, warmup_voice)
            
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")
            
    async def _warmup_inference(self, backend: BaseModelBackend, voice: torch.Tensor) -> None:
        """Run warmup inference to initialize model.
        
        Args:
            backend: Model backend to warm up
            voice: Voice tensor already loaded on correct device
        """
        try:
            # Import here to avoid circular imports
            from ..services.text_processing import process_text
            
            # Use real text
            text = "Testing text to speech synthesis."
            
            # Process through pipeline
            tokens = process_text(text)
            if not tokens:
                raise ValueError("Text processing failed")
            
            # Run inference
            backend.generate(tokens, voice, speed=1.0)
            logger.info("Completed warmup inference")
            
        except Exception as e:
            logger.warning(f"Warmup inference failed: {e}")
            raise

    async def generate(
        self,
        tokens: list[int],
        voice: torch.Tensor,
        speed: float = 1.0,
        backend_type: Optional[str] = None
    ) -> torch.Tensor:
        """Generate audio using specified backend.
        
        Args:
            tokens: Input token IDs
            voice: Voice tensor already loaded on correct device
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
            # Generate audio using provided voice tensor
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