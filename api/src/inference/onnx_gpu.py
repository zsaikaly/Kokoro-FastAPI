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
from ..core.config import settings
from ..structures.model_schemas import ONNXGPUConfig
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
        self._config = ONNXGPUConfig(
            optimization_level=settings.onnx_optimization_level,
            num_threads=settings.onnx_num_threads,
            inter_op_threads=settings.onnx_inter_op_threads,
            execution_mode=settings.onnx_execution_mode,
            memory_pattern=settings.onnx_memory_pattern,
            arena_extend_strategy=settings.onnx_arena_extend_strategy,
            device_id=0,
            gpu_mem_limit=0.7,
            cudnn_conv_algo_search="EXHAUSTIVE",
            do_copy_in_default_stream=True
        )

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
        
        # Set optimization level
        if self._config.optimization_level == "all":
            options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL
        elif self._config.optimization_level == "basic":
            options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_BASIC
        else:
            options.graph_optimization_level = GraphOptimizationLevel.ORT_DISABLE_ALL
        
        # Configure threading
        options.intra_op_num_threads = self._config.num_threads
        options.inter_op_num_threads = self._config.inter_op_threads
        
        # Set execution mode
        options.execution_mode = (
            ExecutionMode.ORT_PARALLEL
            if self._config.execution_mode == "parallel"
            else ExecutionMode.ORT_SEQUENTIAL
        )
        
        # Configure memory optimization
        options.enable_mem_pattern = self._config.memory_pattern
        
        return options

    def _create_provider_options(self) -> Dict:
        """Create CUDA provider options.
        
        Returns:
            Provider configuration
        """
        return {
            "CUDAExecutionProvider": {
                "device_id": self._config.device_id,
                "arena_extend_strategy": self._config.arena_extend_strategy,
                "gpu_mem_limit": int(self._config.gpu_mem_limit * torch.cuda.get_device_properties(0).total_memory),
                "cudnn_conv_algo_search": self._config.cudnn_conv_algo_search,
                "do_copy_in_default_stream": self._config.do_copy_in_default_stream
            }
        }