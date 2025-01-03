import os
import numpy as np
import torch
from loguru import logger
from models import build_model
from kokoro import generate

from .tts_base import TTSBaseModel

class TTSGPUModel(TTSBaseModel):
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
    def generate(cls, input_data: str, voicepack: torch.Tensor, *args) -> np.ndarray:
        """Generate audio using PyTorch model on GPU
        
        Args:
            input_data: Text string to generate audio from
            voicepack: Voice tensor
            *args: (lang, speed) tuple
            
        Returns:
            np.ndarray: Generated audio samples
        """
        if cls._instance is None:
            raise RuntimeError("GPU model not initialized")
            
        lang, speed = args
        result = generate(cls._instance, input_data, voicepack, lang=lang, speed=speed)
        # kokoro.generate returns (audio, metadata, info), we only want audio
        audio = result[0]
        
        # Convert to numpy array if needed
        if isinstance(audio, torch.Tensor):
            audio = audio.cpu().numpy()
        return audio
