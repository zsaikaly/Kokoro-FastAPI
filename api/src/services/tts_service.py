"""TTS service using model and voice managers."""

import io
import os
import time
from typing import List, Tuple

import numpy as np
import scipy.io.wavfile as wavfile
import torch
from loguru import logger

from ..core.config import settings
from ..inference.model_manager import get_manager as get_model_manager
from ..inference.voice_manager import get_manager as get_voice_manager
from .audio import AudioNormalizer, AudioService
from .text_processing import chunker, normalize_text, process_text


class TTSService:
    """Text-to-speech service."""

    def __init__(self, output_dir: str = None):
        """Initialize service.
        
        Args:
            output_dir: Optional output directory for saving audio
        """
        self.output_dir = output_dir
        self.model_manager = get_model_manager()
        self.voice_manager = get_voice_manager()
        self._initialized = False
        self._initialization_error = None

    async def ensure_initialized(self):
        """Ensure model is initialized."""
        if self._initialized:
            return
        if self._initialization_error:
            raise self._initialization_error

        try:
            # Get api directory path (one level up from src)
            api_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            
            # Determine model file and backend based on hardware
            if settings.use_gpu and torch.cuda.is_available():
                model_file = settings.pytorch_model_file
                backend_type = 'pytorch_gpu'
            else:
                model_file = settings.onnx_model_file
                backend_type = 'onnx_cpu'
            
            # Construct model path relative to api directory
            model_path = os.path.join(api_dir, settings.model_dir, model_file)
            
            # Ensure model directory exists
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            
            if not os.path.exists(model_path):
                raise RuntimeError(f"Model file not found: {model_path}")
            
            # Load default voice for warmup
            backend = self.model_manager.get_backend(backend_type)
            warmup_voice = await self.voice_manager.load_voice(settings.default_voice, device=backend.device)
            logger.info(f"Loaded voice {settings.default_voice} for warmup")
            
            # Initialize model with warmup voice
            await self.model_manager.load_model(model_path, warmup_voice, backend_type)
            logger.info(f"Initialized model on {backend_type} backend")
            self._initialized = True
            
        except Exception as e:
            logger.error(f"Failed to initialize model: {e}")
            self._initialization_error = RuntimeError(f"Model initialization failed: {e}")
            raise self._initialization_error

    async def generate_audio(
        self, text: str, voice: str, speed: float = 1.0
    ) -> Tuple[np.ndarray, float]:
        """Generate audio for text.
        
        Args:
            text: Input text
            voice: Voice name
            speed: Speed multiplier
            
        Returns:
            Audio samples and processing time
        """
        await self.ensure_initialized()
        start_time = time.time()

        try:
            # Normalize text
            normalized = normalize_text(text)
            if not normalized:
                raise ValueError("Text is empty after preprocessing")
            text = str(normalized)

            # Process text into chunks
            audio_chunks = []
            for chunk in chunker.split_text(text):
                try:
                    # Convert chunk to token IDs
                    tokens = process_text(chunk)
                    if not tokens:
                        continue

                    # Get backend and load voice
                    backend = self.model_manager.get_backend()
                    voice_tensor = await self.voice_manager.load_voice(voice, device=backend.device)
                    
                    # Generate audio
                    chunk_audio = await self.model_manager.generate(
                        tokens,
                        voice_tensor,
                        speed=speed
                    )
                    if chunk_audio is not None:
                        audio_chunks.append(chunk_audio)
                except Exception as e:
                    logger.error(f"Failed to process chunk: '{chunk}'. Error: {str(e)}")
                    continue

            if not audio_chunks:
                raise ValueError("No audio chunks were generated successfully")

            # Combine chunks
            audio = np.concatenate(audio_chunks) if len(audio_chunks) > 1 else audio_chunks[0]
            processing_time = time.time() - start_time
            return audio, processing_time

        except Exception as e:
            logger.error(f"Error in audio generation: {str(e)}")
            raise

    async def generate_audio_stream(
        self,
        text: str,
        voice: str,
        speed: float = 1.0,
        output_format: str = "wav",
    ):
        """Generate and stream audio chunks.
        
        Args:
            text: Input text
            voice: Voice name
            speed: Speed multiplier
            output_format: Output audio format
            
        Yields:
            Audio chunks as bytes
        """
        await self.ensure_initialized()

        try:
            # Setup audio processing
            stream_normalizer = AudioNormalizer()
            
            # Normalize text
            normalized = normalize_text(text)
            if not normalized:
                raise ValueError("Text is empty after preprocessing")
            text = str(normalized)

            # Process chunks
            is_first = True
            chunk_gen = chunker.split_text(text)
            current_chunk = next(chunk_gen, None)

            while current_chunk is not None:
                next_chunk = next(chunk_gen, None)
                try:
                    # Convert chunk to token IDs
                    tokens = process_text(current_chunk)
                    if tokens:
                        # Get backend and load voice
                        backend = self.model_manager.get_backend()
                        voice_tensor = await self.voice_manager.load_voice(voice, device=backend.device)
                        
                        # Generate audio
                        chunk_audio = await self.model_manager.generate(
                            tokens,
                            voice_tensor,
                            speed=speed
                        )

                        if chunk_audio is not None:
                            # Convert to bytes
                            chunk_bytes = AudioService.convert_audio(
                                chunk_audio,
                                24000,
                                output_format,
                                is_first_chunk=is_first,
                                normalizer=stream_normalizer,
                                is_last_chunk=(next_chunk is None),
                                stream=True
                            )
                            yield chunk_bytes
                            is_first = False

                except Exception as e:
                    logger.error(f"Failed to generate audio for chunk: '{current_chunk}'. Error: {str(e)}")

                current_chunk = next_chunk

        except Exception as e:
            logger.error(f"Error in audio generation stream: {str(e)}")
            raise

    async def combine_voices(self, voices: List[str]) -> str:
        """Combine multiple voices.
        
        Args:
            voices: List of voice names
            
        Returns:
            Name of combined voice
        """
        await self.ensure_initialized()
        return await self.voice_manager.combine_voices(voices)

    async def list_voices(self) -> List[str]:
        """List available voices.
        
        Returns:
            List of voice names
        """
        return await self.voice_manager.list_voices()

    def _audio_to_bytes(self, audio: np.ndarray) -> bytes:
        """Convert audio to WAV bytes.
        
        Args:
            audio: Audio samples
            
        Returns:
            WAV bytes
        """
        buffer = io.BytesIO()
        wavfile.write(buffer, 24000, audio)
        return buffer.getvalue()