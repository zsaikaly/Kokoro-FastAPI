from pydantic import BaseModel, Field
from typing import Literal
from enum import Enum


class TTSStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    DELETED = "deleted"  # For files removed by cleanup


# OpenAI-compatible schemas
class OpenAISpeechRequest(BaseModel):
    model: Literal["tts-1", "tts-1-hd"] = "tts-1"
    input: str = Field(..., description="The text to generate audio for")
    voice: Literal["am_adam", "am_michael", "bm_lewis", "af", "bm_george", "bf_isabella", "bf_emma", "af_sarah", "af_bella"] = Field(
        default="af",
        description="The voice to use for generation"
    )
    response_format: Literal["mp3", "opus", "aac", "flac", "wav", "pcm"] = Field(
        default="mp3",
        description="The format to return audio in. Supported formats: mp3, opus, flac, wav. AAC and PCM are not currently supported."
    )
    speed: float = Field(
        default=1.0,
        ge=0.25,
        le=4.0,
        description="The speed of the generated audio. Select a value from 0.25 to 4.0."
    )
