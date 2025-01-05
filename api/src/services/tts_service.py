import io
import os
import re
import time
from typing import List, Tuple, Optional
from functools import lru_cache

import numpy as np
import torch
import scipy.io.wavfile as wavfile
from .text_processing import normalize_text
from loguru import logger

from ..core.config import settings
from .tts_model import TTSModel
from .audio import AudioService, AudioNormalizer


class TTSService:
    def __init__(self, output_dir: str = None):
        self.output_dir = output_dir

    def _split_text(self, text: str) -> List[str]:
        """Split text into sentences"""
        if not isinstance(text, str):
            text = str(text) if text is not None else ""
        return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]

    @staticmethod
    @lru_cache(maxsize=20)  # Cache up to 8 most recently used voices
    def _load_voice(voice_path: str) -> torch.Tensor:
        """Load and cache a voice model"""
        return torch.load(voice_path, map_location=TTSModel.get_device(), weights_only=True)

    def _get_voice_path(self, voice_name: str) -> Optional[str]:
        """Get the path to a voice file"""
        voice_path = os.path.join(TTSModel.VOICES_DIR, f"{voice_name}.pt")
        return voice_path if os.path.exists(voice_path) else None

    def _generate_audio(
        self, text: str, voice: str, speed: float, stitch_long_output: bool = True
    ) -> Tuple[torch.Tensor, float]:
        """Generate complete audio and return with processing time"""
        audio, processing_time = self._generate_audio_internal(text, voice, speed, stitch_long_output)
        return audio, processing_time

    def _generate_audio_internal(
        self, text: str, voice: str, speed: float, stitch_long_output: bool = True
    ) -> Tuple[torch.Tensor, float]:
        """Generate audio and measure processing time"""
        start_time = time.time()

        try:
            # Normalize text once at the start
            if not text:
                raise ValueError("Text is empty after preprocessing")
            normalized = normalize_text(text)
            if not normalized:
                raise ValueError("Text is empty after preprocessing")
            text = str(normalized)

            # Check voice exists
            voice_path = self._get_voice_path(voice)
            if not voice_path:
                raise ValueError(f"Voice not found: {voice}")

            # Load voice using cached loader
            voicepack = self._load_voice(voice_path)

            # Generate audio with or without stitching
            if stitch_long_output:
                chunks = self._split_text(text)
                audio_chunks = []

                # Process all chunks
                for i, chunk in enumerate(chunks):
                    try:
                        # Process text and generate audio
                        phonemes, tokens = TTSModel.process_text(chunk, voice[0])
                        chunk_audio = TTSModel.generate_from_tokens(tokens, voicepack, speed)
    
                        if chunk_audio is not None:
                            audio_chunks.append(chunk_audio)
                        else:
                            logger.error(f"No audio generated for chunk {i + 1}/{len(chunks)}")
                            
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
                # Process single chunk
                phonemes, tokens = TTSModel.process_text(text, voice[0])
                audio = TTSModel.generate_from_tokens(tokens, voicepack, speed)

            processing_time = time.time() - start_time
            return audio, processing_time

        except Exception as e:
            logger.error(f"Error in audio generation: {str(e)}")
            raise

    async def generate_audio_stream(
        self, text: str, voice: str, speed: float, output_format: str = "wav"
    ):
        """Generate and yield audio chunks as they're generated for real-time streaming"""
        try:
            # Create normalizer for consistent audio levels
            stream_normalizer = AudioNormalizer()
            
            # Input validation and preprocessing
            if not text:
                raise ValueError("Text is empty")
            normalized = normalize_text(text)
            if not normalized:
                raise ValueError("Text is empty after preprocessing")
            text = str(normalized)

            # Voice validation and loading
            voice_path = self._get_voice_path(voice)
            if not voice_path:
                raise ValueError(f"Voice not found: {voice}")
            voicepack = self._load_voice(voice_path)

            # Split text into smaller chunks for faster streaming
            # Use shorter chunks for real-time delivery
            chunks = []
            sentences = self._split_text(text)
            current_chunk = []
            current_length = 0
            target_length = 100  # Target ~100 characters per chunk for faster processing

            for sentence in sentences:
                current_chunk.append(sentence)
                current_length += len(sentence)
                if current_length >= target_length:
                    chunks.append(" ".join(current_chunk))
                    current_chunk = []
                    current_length = 0
            
            if current_chunk:
                chunks.append(" ".join(current_chunk))

            # Process and stream chunks
            for i, chunk in enumerate(chunks):
                try:
                    # Process text and generate audio
                    phonemes, tokens = TTSModel.process_text(chunk, voice[0])
                    chunk_audio = TTSModel.generate_from_tokens(tokens, voicepack, speed)

                    if chunk_audio is not None:
                        # Convert chunk with proper header handling
                        chunk_bytes = AudioService.convert_audio(
                            chunk_audio,
                            24000,
                            output_format,
                            is_first_chunk=(i == 0),
                            normalizer=stream_normalizer
                        )
                        yield chunk_bytes
                    else:
                        logger.error(f"No audio generated for chunk {i + 1}/{len(chunks)}")

                except Exception as e:
                    logger.error(
                        f"Failed to generate audio for chunk {i + 1}/{len(chunks)}: '{chunk}'. Error: {str(e)}"
                    )
                    continue

        except Exception as e:
            logger.error(f"Error in audio generation stream: {str(e)}")
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
        """Combine multiple voices into a new voice"""
        if len(voices) < 2:
            raise ValueError("At least 2 voices are required for combination")

        # Load voices
        t_voices: List[torch.Tensor] = []
        v_name: List[str] = []

        for voice in voices:
            try:
                voice_path = os.path.join(TTSModel.VOICES_DIR, f"{voice}.pt")
                voicepack = torch.load(
                    voice_path, map_location=TTSModel.get_device(), weights_only=True
                )
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
                raise RuntimeError(
                    f"Failed to save combined voice to {combined_path}: {str(e)}"
                )

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
