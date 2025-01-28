"""TTS service using model and voice managers."""

import io
import time
from typing import List, Tuple, Optional, AsyncGenerator, Union

import numpy as np
import scipy.io.wavfile as wavfile
import torch
from loguru import logger

from ..core.config import settings
from ..inference.model_manager import get_manager as get_model_manager
from ..inference.voice_manager import get_manager as get_voice_manager
from .audio import AudioNormalizer, AudioService
from .text_processing import chunker, normalize_text, process_text


import asyncio

class TTSService:
    """Text-to-speech service."""

    # Limit concurrent chunk processing
    _chunk_semaphore = asyncio.Semaphore(4)

    def __init__(self, output_dir: str = None):
        """Initialize service."""
        self.output_dir = output_dir
        self.model_manager = None
        self._voice_manager = None

    @classmethod
    async def create(cls, output_dir: str = None) -> 'TTSService':
        """Create and initialize TTSService instance."""
        service = cls(output_dir)
        service.model_manager = await get_model_manager()
        service._voice_manager = await get_voice_manager()
        return service

    async def _process_chunk(
        self,
        chunk: str,
        voice_tensor: torch.Tensor,
        speed: float,
        output_format: Optional[str] = None,
        is_first: bool = False,
        is_last: bool = False,
        normalizer: Optional[AudioNormalizer] = None,
    ) -> Optional[Union[np.ndarray, bytes]]:
        """Process a single text chunk into audio."""
        async with self._chunk_semaphore:
            try:
                tokens = process_text(chunk)
                if not tokens:
                    return None
                
                # Generate audio using pre-warmed model
                chunk_audio = await self.model_manager.generate(
                    tokens,
                    voice_tensor,
                    speed=speed
                )
                
                if chunk_audio is None:
                    return None
                    
                # For streaming, convert to bytes
                if output_format:
                    return await AudioService.convert_audio(
                        chunk_audio,
                        24000,
                        output_format,
                        is_first_chunk=is_first,
                        normalizer=normalizer,
                        is_last_chunk=is_last
                    )
                    
                return chunk_audio
                
            except Exception as e:
                logger.error(f"Failed to process chunk: '{chunk}'. Error: {str(e)}")
                return None

    async def generate_audio_stream(
        self,
        text: str,
        voice: str,
        speed: float = 1.0,
        output_format: str = "wav",
    ) -> AsyncGenerator[bytes, None]:
        """Generate and stream audio chunks."""
        stream_normalizer = AudioNormalizer()
        voice_tensor = None
        pending_results = {}
        next_index = 0
        
        try:
            # Normalize text
            normalized = normalize_text(text)
            if not normalized:
                raise ValueError("Text is empty after preprocessing")
            text = str(normalized)

            # Get backend and load voice (should be fast if cached)
            backend = self.model_manager.get_backend()
            voice_tensor = await self._voice_manager.load_voice(voice, device=backend.device)

            # Process chunks with semaphore limiting concurrency
            chunks = []
            async for chunk in chunker.split_text(text):
                chunks.append(chunk)
                
            if not chunks:
                raise ValueError("No text chunks to process")

            # Create tasks for all chunks
            tasks = [
                asyncio.create_task(
                    self._process_chunk(
                        chunk,
                        voice_tensor,
                        speed,
                        output_format,
                        is_first=(i == 0),
                        is_last=(i == len(chunks) - 1),
                        normalizer=stream_normalizer
                    )
                )
                for i, chunk in enumerate(chunks)
            ]

            # Process chunks and maintain order
            for i, task in enumerate(tasks):
                result = await task
                
                if i == next_index and result is not None:
                    # If this is the next chunk we need, yield it
                    yield result
                    next_index += 1
                    
                    # Check if we have any subsequent chunks ready
                    while next_index in pending_results:
                        result = pending_results.pop(next_index)
                        if result is not None:
                            yield result
                        next_index += 1
                else:
                    # Store out-of-order result
                    pending_results[i] = result

        except Exception as e:
            logger.error(f"Error in audio generation stream: {str(e)}")
            raise
        finally:
            if voice_tensor is not None:
                del voice_tensor
                torch.cuda.empty_cache()

    async def generate_audio(
        self, text: str, voice: str, speed: float = 1.0
    ) -> Tuple[np.ndarray, float]:
        """Generate complete audio for text using streaming internally."""
        start_time = time.time()
        chunks = []
        
        try:
            # Use streaming generator but collect all chunks
            async for chunk in self.generate_audio_stream(
                text, voice, speed, output_format=None
            ):
                if chunk is not None:
                    chunks.append(chunk)

            if not chunks:
                raise ValueError("No audio chunks were generated successfully")

            # Combine chunks
            audio = np.concatenate(chunks) if len(chunks) > 1 else chunks[0]
            processing_time = time.time() - start_time
            return audio, processing_time

        except Exception as e:
            logger.error(f"Error in audio generation: {str(e)}")
            raise

    async def combine_voices(self, voices: List[str]) -> str:
        """Combine multiple voices."""
        return await self._voice_manager.combine_voices(voices)

    async def list_voices(self) -> List[str]:
        """List available voices."""
        return await self._voice_manager.list_voices()