import os
import numpy as np
import torch
from onnxruntime import InferenceSession, SessionOptions, GraphOptimizationLevel, ExecutionMode
from loguru import logger

class TTSCPUModel:
    _instance = None
    _onnx_session = None

    @classmethod
    def initialize(cls, model_dir: str):
        """Initialize ONNX model for CPU inference"""
        if cls._onnx_session is None:
            # Try loading ONNX model
            onnx_path = os.path.join(model_dir, "kokoro-v0_19.onnx")
            if not os.path.exists(onnx_path):
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
    def generate(cls, tokens: list, voicepack: torch.Tensor, speed: float) -> np.ndarray:
        """Generate audio using ONNX model"""
        if cls._onnx_session is None:
            raise RuntimeError("ONNX model not initialized")

        # Pre-allocate and prepare inputs
        tokens_input = np.array([tokens], dtype=np.int64)
        style_input = voicepack[len(tokens)-2].numpy()  # Already has correct dimensions
        speed_input = np.full(1, speed, dtype=np.float32)  # More efficient than ones * speed
        
        # Run inference with optimized inputs
        return cls._onnx_session.run(
            None,
            {
                'tokens': tokens_input,
                'style': style_input,
                'speed': speed_input
            }
        )[0]
