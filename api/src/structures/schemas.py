from enum import Enum
from typing import Literal

from pydantic import Field, BaseModel


class TTSStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    DELETED = "deleted"  # For files removed by cleanup


# OpenAI-compatible schemas
class OpenAISpeechRequest(BaseModel):
    model: Literal["tts-1", "tts-1-hd", "kokoro"] = "kokoro"
    input: str = Field(..., description="The text to generate audio for")
    voice: str = Field(
        default="af", 
        description="The voice to use for generation. Can be a base voice or a combined voice name."
    )
    response_format: Literal["mp3", "opus", "aac", "flac", "wav", "pcm"] = Field(
        default="mp3",
        description="The format to return audio in. Supported formats: mp3, opus, flac, wav. AAC and PCM are not currently supported.",
    )
    speed: float = Field(
        default=1.0,
        ge=0.25,
        le=4.0,
        description="The speed of the generated audio. Select a value from 0.25 to 4.0.",
    )
