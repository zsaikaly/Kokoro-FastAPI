from datetime import datetime
from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum


class TTSStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class TTSQueueModel(BaseModel):
    id: Optional[int] = None
    text: str
    voice: str = "af"
    speed: float = 1.0
    stitch_long_output: bool = True
    status: TTSStatus = TTSStatus.PENDING
    output_file: Optional[str] = None
    processing_time: Optional[float] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        from_attributes = True


class TTSRequest(BaseModel):
    text: str
    voice: str = "af"  # Default voice
    local: bool = False  # Whether to save file locally or return bytes
    speed: float = Field(default=1.0, gt=0.0, description="Speed multiplier (must be positive)")
    stitch_long_output: bool = True  # Whether to stitch together long outputs


class TTSResponse(BaseModel):
    request_id: int
    status: str
    output_file: Optional[str] = None  # Path for local file
    processing_time: Optional[float] = None  # Processing time in seconds


class VoicesResponse(BaseModel):
    voices: list[str]
    default: str
