"""Audio conversion service"""

from io import BytesIO

import numpy as np
import soundfile as sf
from loguru import logger


class AudioService:
    """Service for audio format conversions"""

    @staticmethod
    def convert_audio(
        audio_data: np.ndarray, sample_rate: int, output_format: str
    ) -> bytes:
        """Convert audio data to specified format

        Args:
            audio_data: Numpy array of audio samples
            sample_rate: Sample rate of the audio
            output_format: Target format (wav, mp3, opus, flac, pcm)

        Returns:
            Bytes of the converted audio
        """
        buffer = BytesIO()

        try:
            if output_format == "wav":
                logger.info("Writing to WAV format...")
                # Ensure audio_data is in int16 format for WAV
                audio_data_wav = (audio_data / np.abs(audio_data).max() * np.iinfo(np.int16).max).astype(np.int16)  # Normalize
                sf.write(buffer, audio_data_wav, sample_rate, format="WAV")
            elif output_format == "mp3":
                logger.info("Converting to MP3 format...")
                # soundfile can write MP3 if ffmpeg or libsox is installed
                sf.write(buffer, audio_data, sample_rate, format="MP3")
            elif output_format == "opus":
                logger.info("Converting to Opus format...")
                sf.write(buffer, audio_data, sample_rate, format="OGG", subtype="OPUS")
            elif output_format == "flac":
                logger.info("Converting to FLAC format...")
                sf.write(buffer, audio_data, sample_rate, format="FLAC")
            elif output_format == "pcm":
                logger.info("Extracting PCM data...")
                # Ensure audio_data is in int16 format for PCM
                audio_data_pcm = (audio_data / np.abs(audio_data).max() * np.iinfo(np.int16).max).astype(np.int16)  # Normalize
                buffer.write(audio_data_pcm.tobytes())
            else:
                raise ValueError(
                    f"Format {output_format} not supported. Supported formats are: wav, mp3, opus, flac, pcm."
                )

            buffer.seek(0)
            return buffer.getvalue()

        except Exception as e:
            logger.error(f"Error converting audio to {output_format}: {str(e)}")
            raise ValueError(f"Failed to convert audio to {output_format}: {str(e)}")
