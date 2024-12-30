import os
import threading
import time
import io
from typing import Optional, Tuple
import torch
import scipy.io.wavfile as wavfile
from models import build_model
from kokoro import generate
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

    def _generate_audio(self, text: str, voice: str) -> Tuple[torch.Tensor, float]:
        """Generate audio and measure processing time"""
        start_time = time.time()

        # Get model instance and voicepack
        model, device = TTSModel.get_instance()
        voicepack = TTSModel.get_voicepack(voice)

        # Generate audio
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
                request_id, text, voice = next_request
                try:
                    # Generate audio and measure time
                    audio, processing_time = self._generate_audio(text, voice)

                    # Save to file
                    output_file = os.path.join(
                        self.output_dir, f"speech_{request_id}.wav"
                    )
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

    def create_tts_request(self, text: str, voice: str = "af") -> int:
        """Create a new TTS request and return the request ID"""
        return self.db.add_request(text, voice)

    def get_request_status(
        self, request_id: int
    ) -> Optional[Tuple[str, Optional[str], Optional[float]]]:
        """Get the status, output file path, and processing time for a request"""
        return self.db.get_status(request_id)
