"""Audio conversion service"""
from io import BytesIO
import numpy as np
import scipy.io.wavfile as wavfile
import soundfile as sf
import logging

logger = logging.getLogger(__name__)

class AudioService:
    """Service for audio format conversions"""
    
    @staticmethod
    def convert_audio(audio_data: np.ndarray, sample_rate: int, output_format: str) -> bytes:
        """Convert audio data to specified format
        
        Args:
            audio_data: Numpy array of audio samples
            sample_rate: Sample rate of the audio
            output_format: Target format (wav, mp3, etc.)
            
        Returns:
            Bytes of the converted audio
        """
        buffer = BytesIO()
        
        try:
            if output_format == 'wav':
                logger.info("Writing to WAV format...")
                wavfile.write(buffer, sample_rate, audio_data)
                return buffer.getvalue()
                
            elif output_format == 'mp3':
                # For MP3, we need to convert to WAV first
                logger.info("Converting to MP3 format...")
                wav_buffer = BytesIO()
                wavfile.write(wav_buffer, sample_rate, audio_data)
                wav_buffer.seek(0)
                
                # Convert WAV to MP3 using soundfile
                buffer = BytesIO()
                sf.write(buffer, audio_data, sample_rate, format='mp3')
                return buffer.getvalue()
                
            elif output_format == 'opus':
                logger.info("Converting to Opus format...")
                sf.write(buffer, audio_data, sample_rate, format='ogg', subtype='opus')
                return buffer.getvalue()
                
            elif output_format == 'flac':
                logger.info("Converting to FLAC format...")
                sf.write(buffer, audio_data, sample_rate, format='flac')
                return buffer.getvalue()
                
            elif output_format == 'aac':
                raise ValueError("AAC format is not currently supported. Please use wav, mp3, opus, or flac.")
                
            elif output_format == 'pcm':
                raise ValueError("PCM format is not currently supported. Please use wav, mp3, opus, or flac.")
                
            else:
                raise ValueError(f"Format {output_format} not supported. Supported formats are: wav, mp3, opus, flac.")
                
        except Exception as e:
            logger.error(f"Error converting audio to {output_format}: {str(e)}")
            raise ValueError(f"Failed to convert audio to {output_format}: {str(e)}")
