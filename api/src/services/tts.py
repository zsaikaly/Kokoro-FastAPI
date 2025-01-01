import io
import os
import re
import threading
import time
from typing import List, Tuple, Optional

import numpy as np
import scipy.io.wavfile as wavfile
import tiktoken
import torch
from loguru import logger

from kokoro import generate, normalize_text, phonemize, tokenize
from models import build_model

from ..core.config import settings

enc = tiktoken.get_encoding("cl100k_base")


class TTSModel:
    _instance = None
    _lock = threading.Lock()
    _voicepacks = {}
    
    # Directory for all voices (copied base voices, and any created combined voices)
    VOICES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "voices")

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
        """Get a voice pack from the voices directory."""
        model, device = cls.get_instance()
        if voice_name not in cls._voicepacks:
            try:
                voice_path = os.path.join(cls.VOICES_DIR, f"{voice_name}.pt")
                if not os.path.exists(voice_path):
                    raise FileNotFoundError(f"Voice file not found: {voice_name}")
                
                voicepack = torch.load(voice_path, map_location=device, weights_only=True)
                cls._voicepacks[voice_name] = voicepack
            except Exception as e:
                logger.error(f"Error loading voice {voice_name}: {str(e)}")
                if voice_name != "af":
                    return cls.get_voicepack("af")
                raise
        return cls._voicepacks[voice_name]


class TTSService:
    def __init__(self, output_dir: str = None, start_worker: bool = False):
        self.output_dir = output_dir
        self._ensure_voices()
        if start_worker:
            self.start_worker()
            
    def _ensure_voices(self):
        """Copy base voices to local voices directory during initialization"""
        os.makedirs(TTSModel.VOICES_DIR, exist_ok=True)
        
        base_voices_dir = os.path.join(settings.model_dir, settings.voices_dir)
        if os.path.exists(base_voices_dir):
            for file in os.listdir(base_voices_dir):
                if file.endswith(".pt"):
                    voice_name = file[:-3]
                    voice_path = os.path.join(TTSModel.VOICES_DIR, file)
                    if not os.path.exists(voice_path):
                        try:
                            base_path = os.path.join(base_voices_dir, file)
                            logger.info(f"Copying base voice {voice_name} to voices directory")
                            voicepack = torch.load(base_path, map_location=TTSModel.get_instance()[1], weights_only=True)
                            torch.save(voicepack, voice_path)
                        except Exception as e:
                            logger.error(f"Error copying voice {voice_name}: {str(e)}")

    def _split_text(self, text: str) -> List[str]:
        """Split text into sentences"""
        return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]

    def _get_voice_path(self, voice_name: str) -> Optional[str]:
        """Get the path to a voice file.
        
        Args:
            voice_name: Name of the voice to find
            
        Returns:
            Path to the voice file if found, None otherwise
        """
        voice_path = os.path.join(TTSModel.VOICES_DIR, f"{voice_name}.pt")
        return voice_path if os.path.exists(voice_path) else None

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

            # Get model instance
            model, device = TTSModel.get_instance()
            
            # Load voice
            voice_path = self._get_voice_path(voice)
            if not voice_path:
                raise ValueError(f"Voice not found: {voice}")
                
            voicepack = torch.load(voice_path, map_location=device, weights_only=True)

            # Generate audio with or without stitching
            if stitch_long_output:
                chunks = self._split_text(text)
                audio_chunks = []

                for i, chunk in enumerate(chunks):
                    try:
                        # Validate phonemization first
                        ps = phonemize(chunk, voice[0])
                        tokens = tokenize(ps)
                        logger.debug(
                            f"Processing chunk {i + 1}/{len(chunks)}: {len(tokens)} tokens"
                        )

                        # Only proceed if phonemization succeeded
                        chunk_audio, _ = generate(
                            model, chunk, voicepack, lang=voice[0], speed=speed
                        )
                        if chunk_audio is not None:
                            audio_chunks.append(chunk_audio)
                        else:
                            logger.error(
                                f"No audio generated for chunk {i + 1}/{len(chunks)}"
                            )
                    except Exception as e:
                        logger.error(
                            f"Failed to generate audio for chunk {i + 1}/{len(chunks)}: '{chunk}'. Error: {str(e)}"
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

    def combine_voices(self, voices: List[str]) -> str:
        """Combine multiple voices into a new voice.
        
        Args:
            voices: List of voice names to combine
            
        Returns:
            Name of the combined voice
            
        Raises:
            ValueError: If less than 2 voices provided or voice loading fails
            RuntimeError: If voice combination or saving fails
        """
        if len(voices) < 2:
            raise ValueError("At least 2 voices are required for combination")
            
        # Load voices
        t_voices: List[torch.Tensor] = []
        v_name: List[str] = []
        
        for voice in voices:
            voice_path = self._get_voice_path(voice)
            if not voice_path:
                raise ValueError(f"Voice not found: {voice}")
                
            try:
                voicepack = torch.load(voice_path, map_location=TTSModel.get_instance()[1], weights_only=True)
                t_voices.append(voicepack)
                v_name.append(voice)
            except Exception as e:
                raise ValueError(f"Failed to load voice {voice}: {str(e)}")
        
        # Combine voices
        try:
            f: str = "_".join(v_name)
            v = torch.mean(torch.stack(t_voices), dim=0)
            combined_path = os.path.join(TTSModel.VOICES_DIR, f"{f}.pt")
            
            # Save combined voice
            try:
                torch.save(v, combined_path)
            except Exception as e:
                raise RuntimeError(f"Failed to save combined voice to {combined_path}: {str(e)}")
                
            return f
            
        except Exception as e:
            if not isinstance(e, (ValueError, RuntimeError)):
                raise RuntimeError(f"Error combining voices: {str(e)}")
            raise

    def list_voices(self) -> List[str]:
        """List all available voices"""
        voices = []
        try:
            for file in os.listdir(TTSModel.VOICES_DIR):
                if file.endswith(".pt"):
                    voices.append(file[:-3])  # Remove .pt extension
        except Exception as e:
            logger.error(f"Error listing voices: {str(e)}")
        return sorted(voices)
