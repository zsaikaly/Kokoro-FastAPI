"""Audio conversion service"""

from io import BytesIO

import numpy as np
import soundfile as sf
from loguru import logger


class AudioNormalizer:
    """Handles audio normalization state for a single stream"""
    def __init__(self):
        self.int16_max = np.iinfo(np.int16).max
    
    def normalize(self, audio_data: np.ndarray) -> np.ndarray:
        """Normalize audio data to int16 range"""
        # Convert to float64 for accurate scaling
        audio_float = audio_data.astype(np.float64)
        
        # Scale to int16 range while preserving relative amplitudes
        max_val = np.abs(audio_float).max()
        if max_val > 0:
            scaling = self.int16_max / max_val
            audio_float *= scaling
        
        # Clip to int16 range and convert
        return np.clip(audio_float, -self.int16_max, self.int16_max).astype(np.int16)

class AudioService:
    """Service for audio format conversions"""
    
    @staticmethod
    def convert_audio(
        audio_data: np.ndarray, 
        sample_rate: int, 
        output_format: str, 
        is_first_chunk: bool = True,
        normalizer: AudioNormalizer = None
    ) -> bytes:
        """Convert audio data to specified format

        Args:
            audio_data: Numpy array of audio samples
            sample_rate: Sample rate of the audio
            output_format: Target format (wav, mp3, opus, flac, pcm)
            is_first_chunk: Whether this is the first chunk of a stream

        Returns:
            Bytes of the converted audio
        """
        buffer = BytesIO()

        try:
            # Normalize audio if normalizer provided, otherwise just convert to int16
            if normalizer is not None:
                normalized_audio = normalizer.normalize(audio_data)
            else:
                normalized_audio = audio_data.astype(np.int16)

            if output_format == "pcm":
                logger.info("Writing PCM data...")
                # Raw 16-bit PCM samples, no header
                buffer.write(normalized_audio.tobytes())
            elif output_format == "wav":
                logger.info("Writing to WAV format...")
                # Always include WAV header for WAV format
                sf.write(buffer, normalized_audio, sample_rate, format="WAV", subtype='PCM_16')
            elif output_format in ["mp3", "aac"]:
                logger.info(f"Converting to {output_format.upper()} format...")
                # Use lower bitrate for streaming
                sf.write(buffer, normalized_audio, sample_rate, format=output_format.upper(), 
                        subtype='COMPRESSED')
            elif output_format == "opus":
                logger.info("Converting to Opus format...")
                # Use lower bitrate and smaller frame size for streaming
                sf.write(buffer, normalized_audio, sample_rate, format="OGG", subtype="OPUS")
            elif output_format == "flac":
                logger.info("Converting to FLAC format...")
                # Use smaller block size for streaming
                sf.write(buffer, normalized_audio, sample_rate, format="FLAC",
                        subtype='PCM_16')
            else:
                raise ValueError(
                    f"Format {output_format} not supported. Supported formats are: wav, mp3, opus, flac, pcm."
                )

            buffer.seek(0)
            return buffer.getvalue()

        except Exception as e:
            logger.error(f"Error converting audio to {output_format}: {str(e)}")
            raise ValueError(f"Failed to convert audio to {output_format}: {str(e)}")
