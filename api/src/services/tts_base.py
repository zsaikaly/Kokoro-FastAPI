import os
import threading
from abc import ABC, abstractmethod
import torch
import numpy as np
from loguru import logger
from kokoro import tokenize, phonemize
from typing import Union, List

from ..core.config import settings


class TTSBaseModel(ABC):
    _instance = None
    _lock = threading.Lock()
    _device = None
    VOICES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "voices")

    @classmethod
    def setup(cls):
        """Initialize model and setup voices"""
        with cls._lock:
            # Set device
            cuda_available = torch.cuda.is_available()
            logger.info(f"CUDA available: {cuda_available}")
            if cuda_available:
                try:
                    # Test CUDA device
                    test_tensor = torch.zeros(1).cuda()
                    logger.info("CUDA test successful")
                    cls._device = "cuda"
                except Exception as e:
                    logger.error(f"CUDA test failed: {e}")
                    cls._device = "cpu"
            else:
                cls._device = "cpu"
            logger.info(f"Initializing model on {cls._device}")

            # Initialize model
            if not cls.initialize(settings.model_dir, settings.model_path):
                raise RuntimeError(f"Failed to initialize {cls._device.upper()} model")

            # Setup voices directory
            os.makedirs(cls.VOICES_DIR, exist_ok=True)

            # Copy base voices to local directory
            base_voices_dir = os.path.join(settings.model_dir, settings.voices_dir)
            if os.path.exists(base_voices_dir):
                for file in os.listdir(base_voices_dir):
                    if file.endswith(".pt"):
                        voice_name = file[:-3]
                        voice_path = os.path.join(cls.VOICES_DIR, file)
                        if not os.path.exists(voice_path):
                            try:
                                logger.info(f"Copying base voice {voice_name} to voices directory")
                                base_path = os.path.join(base_voices_dir, file)
                                voicepack = torch.load(base_path, map_location=cls._device, weights_only=True)
                                torch.save(voicepack, voice_path)
                            except Exception as e:
                                logger.error(f"Error copying voice {voice_name}: {str(e)}")

            # Warm up with default voice
            try:
                dummy_text = "Hello"
                voice_path = os.path.join(cls.VOICES_DIR, "af.pt")
                dummy_voicepack = torch.load(voice_path, map_location=cls._device, weights_only=True)
                
                if cls._device == "cuda":
                    cls.generate(dummy_text, dummy_voicepack, "a", 1.0)
                else:
                    ps = phonemize(dummy_text, "a")
                    tokens = tokenize(ps)
                    tokens = [0] + tokens + [0]
                    cls.generate(tokens, dummy_voicepack, 1.0)
                
                logger.info("Model warm-up complete")
            except Exception as e:
                logger.warning(f"Model warm-up failed: {e}")

            # Count voices in directory
            voice_count = len([f for f in os.listdir(cls.VOICES_DIR) if f.endswith(".pt")])
            return voice_count

    @classmethod
    @abstractmethod
    def initialize(cls, model_dir: str, model_path: str = None):
        """Initialize the model"""
        pass

    @classmethod
    @abstractmethod
    def generate(cls, input_data: Union[str, List[int]], voicepack: torch.Tensor, *args) -> np.ndarray:
        """Generate audio from input
        
        Args:
            input_data: Either text string (GPU) or tokenized input (CPU)
            voicepack: Voice tensor
            *args: Additional args (lang+speed for GPU, speed for CPU)
            
        Returns:
            np.ndarray: Generated audio samples
        """
        pass

    @classmethod
    def get_device(cls):
        """Get the current device"""
        if cls._device is None:
            raise RuntimeError("Model not initialized. Call setup() first.")
        return cls._device
