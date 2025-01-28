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

        # Format-specific setup
        if self.format == "wav":
            self._write_wav_header()
        elif self.format == "ogg":
            self.writer = sf.SoundFile(
                file=BytesIO(),
                mode='w',
                samplerate=sample_rate,
                channels=channels,
                format='OGG',
                subtype='VORBIS'
            )
        elif self.format == "mp3":
            # For MP3, we'll use pydub's incremental writer
            self.buffer = BytesIO()
            self.segments = []  # Store segments until we have enough data
            # Initialize an empty AudioSegment as our encoder
            self.encoder = AudioSegment.silent(duration=0, frame_rate=self.sample_rate)

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
        buffer = BytesIO()

        if finalize:
            if self.format == "wav":
                # Write final WAV header with correct sizes
                buffer.write(b'RIFF')
                buffer.write(struct.pack('<L', self.bytes_written + 36))
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
            elif self.format == "ogg":
                self.writer.close()
            elif self.format == "mp3":
                # Final export of any remaining audio
                if hasattr(self, 'encoder') and len(self.encoder) > 0:
                    self.encoder.export(buffer, format="mp3", bitrate="192k", parameters=["-q:a", "2"])
                    self.encoder = None
            return buffer.getvalue()

        if audio_data is None or len(audio_data) == 0:
            return b''

        if self.format == "wav":
            # For WAV, write raw PCM after the first chunk
            if self.bytes_written == 0:
                buffer.write(self._write_wav_header())
            buffer.write(audio_data.tobytes())
            self.bytes_written += len(audio_data.tobytes())

        elif self.format == "ogg":
            # OGG/Vorbis handles streaming naturally
            self.writer.write(audio_data)
            self.writer.flush()
            buffer = self.writer.file
            buffer.seek(0, 2)  # Seek to end
            chunk = buffer.getvalue()
            buffer.seek(0)
            buffer.truncate()
            return chunk

        elif self.format == "mp3":
            # Convert chunk to AudioSegment and encode
            segment = AudioSegment(
                audio_data.tobytes(),
                frame_rate=self.sample_rate,
                sample_width=audio_data.dtype.itemsize,
                channels=self.channels
            )
            
            # Add segment to encoder
            self.encoder = self.encoder + segment
            
            # Export current state to buffer
            self.encoder.export(buffer, format="mp3", bitrate="192k", parameters=["-q:a", "2"])
            
            # Get the encoded data
            encoded_data = buffer.getvalue()
            
            # Reset encoder to prevent memory growth
            self.encoder = AudioSegment.silent(duration=0, frame_rate=self.sample_rate)
            
            return encoded_data

        return buffer.getvalue()

    def close(self) -> Optional[bytes]:
        """Finish the audio file and return any remaining data"""
        if self.format == "wav":
            # Update WAV header with final file size
            buffer = BytesIO()
            buffer.write(b'RIFF')
            buffer.write(struct.pack('<L', self.bytes_written + 36))  # File size
            buffer.write(b'WAVE')
            # ... rest of header ...
            buffer.write(struct.pack('<L', self.bytes_written))  # Data size
            return buffer.getvalue()

        elif self.format == "ogg":
            self.writer.close()
            return None

        elif self.format == "mp3":
            # Flush any remaining MP3 frames
            buffer = BytesIO()
            self.encoder.export(buffer, format="mp3")
            return buffer.getvalue()