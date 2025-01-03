import os
import threading
import torch
from loguru import logger
from kokoro import tokenize, phonemize

from ..core.config import settings
from .tts_cpu import TTSCPUModel
from .tts_gpu import TTSGPUModel


class TTSModel:
    _device = None
    _lock = threading.Lock()
    VOICES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "voices")

    @classmethod
    def initialize(cls):
        """Initialize and warm up the model"""
        with cls._lock:
            # Set device and initialize model
            cls._device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Initializing model on {cls._device}")

            # Initialize appropriate model based on device
            if cls._device == "cuda":
                if not TTSGPUModel.initialize(settings.model_dir, settings.model_path):
                    raise RuntimeError("Failed to initialize GPU model")
            else:
                # Try CPU ONNX first, fallback to CPU PyTorch if needed
                if not TTSCPUModel.initialize(settings.model_dir):
                    logger.warning("ONNX initialization failed, falling back to PyTorch CPU")
                    if not TTSGPUModel.initialize(settings.model_dir, settings.model_path):
                        raise RuntimeError("Failed to initialize CPU model")

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
                                logger.info(
                                    f"Copying base voice {voice_name} to voices directory"
                                )
                                base_path = os.path.join(base_voices_dir, file)
                                voicepack = torch.load(
                                    base_path,
                                    map_location=cls._device,
                                    weights_only=True,
                                )
                                torch.save(voicepack, voice_path)
                            except Exception as e:
                                logger.error(
                                    f"Error copying voice {voice_name}: {str(e)}"
                                )

            # Warm up with default voice
            try:
                dummy_text = "Hello"
                voice_path = os.path.join(cls.VOICES_DIR, "af.pt")
                dummy_voicepack = torch.load(
                    voice_path, map_location=cls._device, weights_only=True
                )
                
                if cls._device == "cuda":
                    TTSGPUModel.generate(dummy_text, dummy_voicepack, "a", 1.0)
                else:
                    ps = phonemize(dummy_text, "a")
                    tokens = tokenize(ps)
                    tokens = [0] + tokens + [0]
                    TTSCPUModel.generate(tokens, dummy_voicepack, 1.0)
                
                logger.info("Model warm-up complete")
            except Exception as e:
                logger.warning(f"Model warm-up failed: {e}")

            # Count voices in directory
            voice_count = len(
                [f for f in os.listdir(cls.VOICES_DIR) if f.endswith(".pt")]
            )
            return voice_count

    @classmethod
    def get_device(cls):
        """Get the current device or raise an error"""
        if cls._device is None:
            raise RuntimeError("Model not initialized. Call initialize() first.")
        return cls._device
