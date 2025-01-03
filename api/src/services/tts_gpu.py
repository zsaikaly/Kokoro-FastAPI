import os
import torch
from loguru import logger
from models import build_model
from kokoro import generate

class TTSGPUModel:
    _instance = None
    _device = "cuda"

    @classmethod
    def initialize(cls, model_dir: str, model_path: str):
        """Initialize PyTorch model for GPU inference"""
        if cls._instance is None and torch.cuda.is_available():
            try:
                logger.info("Initializing GPU model")
                model_path = os.path.join(model_dir, model_path)
                model = build_model(model_path, cls._device)
                cls._instance = model
                return cls._instance
            except Exception as e:
                logger.error(f"Failed to initialize GPU model: {e}")
                return None
        return cls._instance

    @classmethod
    def generate(cls, text: str, voicepack: torch.Tensor, lang: str, speed: float) -> tuple[torch.Tensor, dict]:
        """Generate audio using PyTorch model on GPU"""
        if cls._instance is None:
            raise RuntimeError("GPU model not initialized")
            
        return generate(cls._instance, text, voicepack, lang=lang, speed=speed)
