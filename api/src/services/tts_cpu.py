import os
import numpy as np
import torch
from onnxruntime import InferenceSession, SessionOptions, GraphOptimizationLevel, ExecutionMode
from loguru import logger

from .tts_base import TTSBaseModel

class TTSCPUModel(TTSBaseModel):
    _instance = None
    _onnx_session = None

    @classmethod
    def initialize(cls, model_dir: str, model_path: str = None):
        """Initialize ONNX model for CPU inference"""
        if cls._onnx_session is None:
            # Try loading ONNX model
            # First try the specified path if provided
            if model_path and model_path.endswith('.onnx'):
                onnx_path = os.path.join(model_dir, model_path)
                if os.path.exists(onnx_path):
                    logger.info(f"Loading specified ONNX model from {onnx_path}")
                else:
                    onnx_path = None
            else:
                # Look for any .onnx file in the directory as fallback
                onnx_files = [f for f in os.listdir(model_dir) if f.endswith('.onnx')]
                if onnx_files:
                    onnx_path = os.path.join(model_dir, onnx_files[0])
                    logger.info(f"Found ONNX model: {onnx_path}")
                else:
                    logger.error(f"No ONNX model found in {model_dir}")
                    return None

            if not onnx_path:
                return None

            logger.info(f"Loading ONNX model from {onnx_path}")
            
            # Configure ONNX session for optimal performance
            session_options = SessionOptions()
            session_options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL
            session_options.intra_op_num_threads = 4  # Adjust based on CPU cores
            session_options.execution_mode = ExecutionMode.ORT_SEQUENTIAL

            # Configure CPU provider options
            provider_options = {
                'CPUExecutionProvider': {
                    'arena_extend_strategy': 'kNextPowerOfTwo',
                    'cpu_memory_arena_cfg': 'cpu:0'
                }
            }

            cls._onnx_session = InferenceSession(
                onnx_path,
                sess_options=session_options,
                providers=['CPUExecutionProvider'],
                provider_options=[provider_options]
            )
            
            return cls._onnx_session
        return cls._onnx_session

    @classmethod
    def generate(cls, input_data: list[int], voicepack: torch.Tensor, *args) -> np.ndarray:
        """Generate audio using ONNX model
        
        Args:
            input_data: list of token IDs
            voicepack: Voice tensor
            *args: (speed,) tuple
            
        Returns:
            np.ndarray: Generated audio samples
        """
        if cls._onnx_session is None:
            raise RuntimeError("ONNX model not initialized")

        speed = args[0]
        # Pre-allocate and prepare inputs
        tokens_input = np.array([input_data], dtype=np.int64)
        style_input = voicepack[len(input_data)-2].numpy()  # Already has correct dimensions
        speed_input = np.full(1, speed, dtype=np.float32)  # More efficient than ones * speed
        
        # Run inference with optimized inputs
        result = cls._onnx_session.run(
            None,
            {
                'tokens': tokens_input,
                'style': style_input,
                'speed': speed_input
            }
        )
        return result[0]
