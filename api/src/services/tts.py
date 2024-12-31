import io
import os
import re
import time
import threading
from typing import List, Tuple

import numpy as np
import torch
import tiktoken
import scipy.io.wavfile as wavfile
from kokoro import generate, tokenize, phonemize, normalize_text
from loguru import logger
from models import build_model

from ..core.config import settings

enc = tiktoken.get_encoding("cl100k_base")


class TTSModel:
    _instance = None
    _lock = threading.Lock()
    _voicepacks = {}

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                    print(f"Initializing model on {device}")
                    model_path = os.path.join(settings.model_dir, settings.model_path)
                    model = build_model(model_path, device)
                    # Note: RNN memory optimization is handled internally by the model
                    cls._instance = (model, device)
        return cls._instance

    @classmethod
    def get_voicepack(cls, voice_name: str) -> torch.Tensor:
        model, device = cls.get_instance()
        if voice_name not in cls._voicepacks:
            try:
                voice_path = os.path.join(
                    settings.model_dir, settings.voices_dir, f"{voice_name}.pt"
                )
                voicepack = torch.load(
                    voice_path, map_location=device, weights_only=True
                )
                cls._voicepacks[voice_name] = voicepack
            except Exception as e:
                print(f"Error loading voice {voice_name}: {str(e)}")
                if voice_name != "af":
                    return cls.get_voicepack("af")
                raise
        return cls._voicepacks[voice_name]


class TTSService:
    def __init__(self, output_dir: str = None, start_worker: bool = False):
        self.output_dir = output_dir
        if start_worker:
            self.start_worker()

    def _split_text(self, text: str) -> List[str]:
        """Split text into sentences"""
        return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]

    def _generate_audio(
        self, text: str, voice: str, speed: float, stitch_long_output: bool = True
    ) -> Tuple[torch.Tensor, float]:
        """Generate audio and measure processing time"""
        start_time = time.time()

        try:
            # Normalize text once at the start
            text = normalize_text(text)
            if not text:
                raise ValueError("Text is empty after preprocessing")

            # Get model instance and voicepack
            model, device = TTSModel.get_instance()
            voicepack = TTSModel.get_voicepack(voice)

            # Generate audio with or without stitching
            if stitch_long_output:
                chunks = self._split_text(text)
                audio_chunks = []

                for i, chunk in enumerate(chunks):
                    try:
                        # Validate phonemization first
                        ps = phonemize(chunk, voice[0])
                        tokens = tokenize(ps)
                        logger.info(
                            f"Processing chunk {i+1}/{len(chunks)}: {len(tokens)} tokens"
                        )

                        # Only proceed if phonemization succeeded
                        chunk_audio, _ = generate(
                            model, chunk, voicepack, lang=voice[0], speed=speed
                        )
                        if chunk_audio is not None:
                            audio_chunks.append(chunk_audio)
                        else:
                            logger.error(
                                f"No audio generated for chunk {i+1}/{len(chunks)}"
                            )
                    except Exception as e:
                        logger.error(
                            f"Failed to generate audio for chunk {i+1}/{len(chunks)}: '{chunk}'. Error: {str(e)}"
                        )
                        continue

                if not audio_chunks:
                    raise ValueError("No audio chunks were generated successfully")

                audio = (
                    np.concatenate(audio_chunks)
                    if len(audio_chunks) > 1
                    else audio_chunks[0]
                )
            else:
                audio, _ = generate(model, text, voicepack, lang=voice[0], speed=speed)

            processing_time = time.time() - start_time
            return audio, processing_time

        except Exception as e:
            print(f"Error in audio generation: {str(e)}")
            raise

    def _save_audio(self, audio: torch.Tensor, filepath: str):
        """Save audio to file"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        wavfile.write(filepath, 24000, audio)

    def _audio_to_bytes(self, audio: torch.Tensor) -> bytes:
        """Convert audio tensor to WAV bytes"""
        buffer = io.BytesIO()
        wavfile.write(buffer, 24000, audio)
        return buffer.getvalue()

    def list_voices(self) -> list[str]:
        """List all available voices"""
        voices = []
        try:
            voices_path = os.path.join(settings.model_dir, settings.voices_dir)
            for file in os.listdir(voices_path):
                if file.endswith(".pt"):
                    voice_name = file[:-3]  # Remove .pt extension
                    voices.append(voice_name)
        except Exception as e:
            print(f"Error listing voices: {str(e)}")
        return voices
