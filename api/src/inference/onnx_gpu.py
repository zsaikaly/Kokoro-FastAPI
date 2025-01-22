"""GPU-based ONNX inference backend."""

from typing import Dict, Optional

import numpy as np
import torch
from loguru import logger
from onnxruntime import (
    ExecutionMode,
    GraphOptimizationLevel,
    InferenceSession,
    SessionOptions
)

from ..core import paths
from ..core.model_config import model_config
from .base import BaseModelBackend


class ONNXGPUBackend(BaseModelBackend):
    """ONNX-based GPU inference backend."""

    def __init__(self):
        """Initialize GPU backend."""
        super().__init__()
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available")
        self._device = "cuda"
        self._session: Optional[InferenceSession] = None

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._session is not None

    async def load_model(self, path: str) -> None:
        """Load ONNX model.
        
        Args:
            path: Path to model file
            
        Raises:
            RuntimeError: If model loading fails
        """
        try:
            # Get verified model path
            model_path = await paths.get_model_path(path)
            
            logger.info(f"Loading ONNX model on GPU: {model_path}")
            
            # Configure session
            options = self._create_session_options()
            provider_options = self._create_provider_options()
            
            # Create session with CUDA provider
            self._session = InferenceSession(
                model_path,
                sess_options=options,
                providers=["CUDAExecutionProvider"],
                provider_options=[provider_options]
            )
            
        except Exception as e:
            raise RuntimeError(f"Failed to load ONNX model: {e}")

    def generate(
        self,
        tokens: list[int],
        voice: torch.Tensor,
        speed: float = 1.0
    ) -> np.ndarray:
        """Generate audio using ONNX model.
        
        Args:
            tokens: Input token IDs
            voice: Voice embedding tensor
            speed: Speed multiplier
            
        Returns:
            Generated audio samples
            
        Raises:
            RuntimeError: If generation fails
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")

        try:
            # Prepare inputs
            tokens_input = np.array([tokens], dtype=np.int64)
            style_input = voice[len(tokens)].cpu().numpy()  # Move to CPU for ONNX
            speed_input = np.full(1, speed, dtype=np.float32)

            # Run inference
            result = self._session.run(
                None,
                {
                    "tokens": tokens_input,
                    "style": style_input,
                    "speed": speed_input
                }
            )
            
            return result[0]
            
        except Exception as e:
            raise RuntimeError(f"Generation failed: {e}")

    def _create_session_options(self) -> SessionOptions:
        """Create ONNX session options.
        
        Returns:
            Configured session options
        """
        options = SessionOptions()
        config = model_config.onnx_gpu
        
        # Set optimization level
        if config.optimization_level == "all":
            options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL
        elif config.optimization_level == "basic":
            options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_BASIC
        else:
            options.graph_optimization_level = GraphOptimizationLevel.ORT_DISABLE_ALL
        
        # Configure threading
        options.intra_op_num_threads = config.num_threads
        options.inter_op_num_threads = config.inter_op_threads
        
        # Set execution mode
        options.execution_mode = (
            ExecutionMode.ORT_PARALLEL
            if config.execution_mode == "parallel"
            else ExecutionMode.ORT_SEQUENTIAL
        )
        
        # Configure memory optimization
        options.enable_mem_pattern = config.memory_pattern
        
        return options

    def _create_provider_options(self) -> Dict:
        """Create CUDA provider options.
        
        Returns:
            Provider configuration
        """
        config = model_config.onnx_gpu
        return {
            "CUDAExecutionProvider": {
                "device_id": config.device_id,
                "arena_extend_strategy": config.arena_extend_strategy,
                "gpu_mem_limit": int(config.gpu_mem_limit * torch.cuda.get_device_properties(0).total_memory),
                "cudnn_conv_algo_search": config.cudnn_conv_algo_search,
                "do_copy_in_default_stream": config.do_copy_in_default_stream
            }
        }

    def unload(self) -> None:
        """Unload model and free resources."""
        if self._session is not None:
            del self._session
            self._session = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()