"""Audio conversion service with proper streaming support"""

from io import BytesIO
import struct
from typing import Generator, Optional

import numpy as np
import soundfile as sf
from loguru import logger
from pydub import AudioSegment

class StreamingAudioWriter:
    """Handles streaming audio format conversions"""

    def __init__(self, format: str, sample_rate: int, channels: int = 1):
        self.format = format.lower()
        self.sample_rate = sample_rate
        self.channels = channels
        self.bytes_written = 0
        self.buffer = BytesIO()

        # Format-specific setup
        if self.format == "wav":
            self._write_wav_header()
        elif self.format in ["ogg", "opus"]:
            # For OGG/Opus, write to memory buffer
            self.writer = sf.SoundFile(
                file=self.buffer,
                mode='w',
                samplerate=sample_rate,
                channels=channels,
                format='OGG',
                subtype='VORBIS' if self.format == "ogg" else "OPUS"
            )
        elif self.format == "flac":
            # For FLAC, write to memory buffer
            self.writer = sf.SoundFile(
                file=self.buffer,
                mode='w',
                samplerate=sample_rate,
                channels=channels,
                format='FLAC'
            )
        elif self.format in ["mp3", "aac"]:
            # For MP3/AAC, we'll use pydub's incremental writer
            self.segments = []  # Store segments until we have enough data
            self.total_duration = 0  # Track total duration in milliseconds
            # Initialize an empty AudioSegment as our encoder
            self.encoder = AudioSegment.silent(duration=0, frame_rate=self.sample_rate)
        elif self.format == "pcm":
            # PCM doesn't need initialization, we'll write raw bytes
            pass
        else:
            raise ValueError(f"Unsupported format: {format}")

    def _write_wav_header(self) -> bytes:
        """Write WAV header with correct streaming format"""
        header = BytesIO()
        header.write(b'RIFF')
        header.write(struct.pack('<L', 0))  # Placeholder for file size
        header.write(b'WAVE')
        header.write(b'fmt ')
        header.write(struct.pack('<L', 16))  # fmt chunk size
        header.write(struct.pack('<H', 1))   # PCM format
        header.write(struct.pack('<H', self.channels))
        header.write(struct.pack('<L', self.sample_rate))
        header.write(struct.pack('<L', self.sample_rate * self.channels * 2))  # Byte rate
        header.write(struct.pack('<H', self.channels * 2))  # Block align
        header.write(struct.pack('<H', 16))  # Bits per sample
        header.write(b'data')
        header.write(struct.pack('<L', 0))  # Placeholder for data size
        return header.getvalue()

    def write_chunk(self, audio_data: Optional[np.ndarray] = None, finalize: bool = False) -> bytes:
        """Write a chunk of audio data and return bytes in the target format.
        
        Args:
            audio_data: Audio data to write, or None if finalizing
            finalize: Whether this is the final write to close the stream
        """
        output_buffer = BytesIO()

        if finalize:
            if self.format == "wav":
                # Write final WAV header with correct sizes
                output_buffer.write(b'RIFF')
                output_buffer.write(struct.pack('<L', self.bytes_written + 36))
                output_buffer.write(b'WAVE')
                output_buffer.write(b'fmt ')
                output_buffer.write(struct.pack('<L', 16))
                output_buffer.write(struct.pack('<H', 1))
                output_buffer.write(struct.pack('<H', self.channels))
                output_buffer.write(struct.pack('<L', self.sample_rate))
                output_buffer.write(struct.pack('<L', self.sample_rate * self.channels * 2))
                output_buffer.write(struct.pack('<H', self.channels * 2))
                output_buffer.write(struct.pack('<H', 16))
                output_buffer.write(b'data')
                output_buffer.write(struct.pack('<L', self.bytes_written))
            elif self.format in ["ogg", "opus", "flac"]:
                self.writer.close()
                return self.buffer.getvalue()
            elif self.format in ["mp3", "aac"]:
                # Final export of any remaining audio
                if hasattr(self, 'encoder') and len(self.encoder) > 0:
                    # Export with duration metadata
                    format_args = {
                        "mp3": {"format": "mp3", "codec": "libmp3lame"},
                        "aac": {"format": "adts", "codec": "aac"}
                    }[self.format]
                    
                    self.encoder.export(
                        output_buffer,
                        **format_args,
                        bitrate="192k",
                        parameters=[
                            "-q:a", "2",
                            "-write_xing", "1" if self.format == "mp3" else "0",  # XING header for MP3 only
                            "-metadata", f"duration={self.total_duration/1000}"  # Duration in seconds
                        ]
                    )
                    self.encoder = None
            return output_buffer.getvalue()

        if audio_data is None or len(audio_data) == 0:
            return b''

        if self.format == "wav":
            # For WAV, write raw PCM after the first chunk
            if self.bytes_written == 0:
                output_buffer.write(self._write_wav_header())
            output_buffer.write(audio_data.tobytes())
            self.bytes_written += len(audio_data.tobytes())

        elif self.format in ["ogg", "opus", "flac"]:
            # Write to soundfile buffer
            self.writer.write(audio_data)
            self.writer.flush()
            # Get current buffer contents
            data = self.buffer.getvalue()
            # Clear buffer for next chunk
            self.buffer.seek(0)
            self.buffer.truncate()
            return data

        elif self.format in ["mp3", "aac"]:
            # Convert chunk to AudioSegment and encode
            segment = AudioSegment(
                audio_data.tobytes(),
                frame_rate=self.sample_rate,
                sample_width=audio_data.dtype.itemsize,
                channels=self.channels
            )
            
            # Track total duration
            self.total_duration += len(segment)
            
            # Add segment to encoder
            self.encoder = self.encoder + segment
            
            # Export current state to buffer
            format_args = {
                "mp3": {"format": "mp3", "codec": "libmp3lame"},
                "aac": {"format": "adts", "codec": "aac"}
            }[self.format]
            
            self.encoder.export(output_buffer, **format_args, bitrate="192k", parameters=[
                "-q:a", "2",
                "-write_xing", "1" if self.format == "mp3" else "0",  # XING header for MP3 only
                "-metadata", f"duration={self.total_duration/1000}"  # Duration in seconds
            ])
            
            # Get the encoded data
            encoded_data = output_buffer.getvalue()
            
            # Reset encoder to prevent memory growth
            self.encoder = AudioSegment.silent(duration=0, frame_rate=self.sample_rate)
            
            return encoded_data

        elif self.format == "pcm":
            # For PCM, just write raw bytes
            return audio_data.tobytes()

        return output_buffer.getvalue()

    def close(self) -> Optional[bytes]:
        """Finish the audio file and return any remaining data"""
        if self.format == "wav":
            # Update WAV header with final file size
            buffer = BytesIO()
            buffer.write(b'RIFF')
            buffer.write(struct.pack('<L', self.bytes_written + 36))  # File size
            buffer.write(b'WAVE')
            buffer.write(b'fmt ')
            buffer.write(struct.pack('<L', 16))
            buffer.write(struct.pack('<H', 1))
            buffer.write(struct.pack('<H', self.channels))
            buffer.write(struct.pack('<L', self.sample_rate))
            buffer.write(struct.pack('<L', self.sample_rate * self.channels * 2))
            buffer.write(struct.pack('<H', self.channels * 2))
            buffer.write(struct.pack('<H', 16))
            buffer.write(b'data')
            buffer.write(struct.pack('<L', self.bytes_written))
            return buffer.getvalue()

        elif self.format in ["ogg", "opus", "flac"]:
            self.writer.close()
            return self.buffer.getvalue()

        elif self.format in ["mp3", "aac"]:
            # Flush any remaining audio
            buffer = BytesIO()
            if hasattr(self, 'encoder') and len(self.encoder) > 0:
                format_args = {
                    "mp3": {"format": "mp3", "codec": "libmp3lame"},
                    "aac": {"format": "adts", "codec": "aac"}
                }[self.format]
                self.encoder.export(buffer, **format_args)
            return buffer.getvalue()

        return None