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
            self.encoder = AudioSegment.from_mono_audiosegments()

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

    def write_chunk(self, audio_data: np.ndarray) -> bytes:
        """Write a chunk of audio data and return bytes in the target format"""
        buffer = BytesIO()

        if self.format == "wav":
            # For WAV, we write raw PCM after the first chunk
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
            self.encoder += segment
            self.encoder.export(buffer, format="mp3")

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