import os
import threading
import time
import io
from typing import Optional, Tuple, List
import numpy as np
import torch
import scipy.io.wavfile as wavfile
from models import build_model
from kokoro import generate, phonemize, tokenize
from ..database.queue import QueueDB


class TTSModel:
    _instance = None
    _lock = threading.Lock()
    _voicepacks = {}

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                    print(f"Initializing model on {device}")
                    model = build_model("kokoro-v0_19.pth", device)
                    cls._instance = (model, device)
        return cls._instance

    @classmethod
    def get_voicepack(cls, voice_name: str) -> torch.Tensor:
        model, device = cls.get_instance()
        if voice_name not in cls._voicepacks:
            try:
                voicepack = torch.load(
                    f"voices/{voice_name}.pt", map_location=device, weights_only=True
                )
                cls._voicepacks[voice_name] = voicepack
            except Exception as e:
                print(f"Error loading voice {voice_name}: {str(e)}")
                if voice_name != "af":
                    return cls.get_voicepack("af")
                raise
        return cls._voicepacks[voice_name]


class TTSService:
    def __init__(self, output_dir: str = None):
        if output_dir is None:
            output_dir = os.path.join(os.path.dirname(__file__), "..", "output")
        self.output_dir = output_dir
        self.db = QueueDB()
        os.makedirs(output_dir, exist_ok=True)
        self._start_worker()

    def _start_worker(self):
        """Start the background worker thread"""
        self.worker = threading.Thread(target=self._process_queue, daemon=True)
        self.worker.start()

    def _find_boundary(self, text: str, max_tokens: int, voice: str, margin: int = 50) -> int:
        """Find the closest sentence/clause boundary within token limit"""
        # Try different boundary markers in order of preference
        for marker in ['. ', '; ', ', ']:
            # Look for the last occurrence of marker before max_tokens
            test_text = text[:max_tokens + margin]  # Look a bit beyond the limit
            last_idx = test_text.rfind(marker)
            
            if last_idx != -1:
                # Verify this boundary is within our token limit
                candidate = text[:last_idx + len(marker)].strip()
                ps = phonemize(candidate, voice[0])
                tokens = tokenize(ps)
                
                if len(tokens) <= max_tokens:
                    return last_idx + len(marker)
        
        # If no good boundary found, find last whitespace within limit
        test_text = text[:max_tokens]
        last_space = test_text.rfind(' ')
        return last_space if last_space != -1 else max_tokens

    def _split_text(self, text: str, voice: str) -> List[str]:
        """Split text into chunks that respect token limits and try to maintain sentence structure"""
        MAX_TOKENS = 450  # Leave wider margin from 510 limit to account for tokenizer differences
        chunks = []
        remaining = text
        
        while remaining:
            # If remaining text is within limit, add it as final chunk
            ps = phonemize(remaining, voice[0])
            tokens = tokenize(ps)
            if len(tokens) <= MAX_TOKENS:
                chunks.append(remaining.strip())
                break
            
            # Find best boundary position
            split_pos = self._find_boundary(remaining, MAX_TOKENS, voice)
            
            # Add chunk and continue with remaining text
            chunks.append(remaining[:split_pos].strip())
            remaining = remaining[split_pos:].strip()
        
        return chunks

    def _generate_audio(self, text: str, voice: str, stitch_long_output: bool = True) -> Tuple[torch.Tensor, float]:
        """Generate audio and measure processing time"""
        start_time = time.time()

        # Get model instance and voicepack
        model, device = TTSModel.get_instance()
        voicepack = TTSModel.get_voicepack(voice)

        # Generate audio with or without stitching
        if stitch_long_output:
            # Split text if needed and generate audio for each chunk
            chunks = self._split_text(text, voice)
            audio_chunks = []
            
            for chunk in chunks:
                chunk_audio, _ = generate(model, chunk, voicepack, lang=voice[0])
                audio_chunks.append(chunk_audio)
            
            # Concatenate audio chunks
            if len(audio_chunks) > 1:
                audio = np.concatenate(audio_chunks)
            else:
                audio = audio_chunks[0]
        else:
            # Generate single chunk without splitting
            audio, _ = generate(model, text, voicepack, lang=voice[0])

        processing_time = time.time() - start_time
        return audio, processing_time

    def _save_audio(self, audio: torch.Tensor, filepath: str):
        """Save audio to file"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        wavfile.write(filepath, 24000, audio)

    def _audio_to_bytes(self, audio: torch.Tensor) -> bytes:
        """Convert audio tensor to WAV bytes"""
        buffer = io.BytesIO()
        wavfile.write(buffer, 24000, audio)
        return buffer.getvalue()

    def _process_queue(self):
        """Background worker that processes the queue"""
        while True:
            next_request = self.db.get_next_pending()
            if next_request:
                request_id, text, voice, stitch_long_output = next_request
                try:
                    # Generate audio and measure time
                    audio, processing_time = self._generate_audio(text, voice, stitch_long_output)

                    # Save to file
                    output_file = os.path.abspath(os.path.join(
                        self.output_dir, f"speech_{request_id}.wav"
                    ))
                    self._save_audio(audio, output_file)

                    # Update status with processing time
                    self.db.update_status(
                        request_id,
                        "completed",
                        output_file=output_file,
                        processing_time=processing_time,
                    )

                except Exception as e:
                    print(f"Error processing request {request_id}: {str(e)}")
                    self.db.update_status(request_id, "failed")

            time.sleep(1)  # Prevent busy waiting

    def list_voices(self) -> list[str]:
        """List all available voices"""
        voices = []
        try:
            for file in os.listdir("voices"):
                if file.endswith(".pt"):
                    voice_name = file[:-3]  # Remove .pt extension
                    voices.append(voice_name)
        except Exception as e:
            print(f"Error listing voices: {str(e)}")
        return voices

    def create_tts_request(self, text: str, voice: str = "af", stitch_long_output: bool = True) -> int:
        """Create a new TTS request and return the request ID"""
        return self.db.add_request(text, voice, stitch_long_output)

    def get_request_status(
        self, request_id: int
    ) -> Optional[Tuple[str, Optional[str], Optional[float]]]:
        """Get the status, output file path, and processing time for a request"""
        return self.db.get_status(request_id)
