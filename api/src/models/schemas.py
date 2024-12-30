from pydantic import BaseModel

from typing import Optional


class TTSRequest(BaseModel):
    text: str
    voice: str = "af"  # Default voice
    local: bool = False  # Whether to save file locally or return bytes
    stitch_long_output: bool = True  # Whether to stitch together long outputs


class TTSResponse(BaseModel):
    request_id: int
    status: str
    output_file: Optional[str] = None  # Path for local file
    processing_time: Optional[float] = None  # Processing time in seconds


class VoicesResponse(BaseModel):
    voices: list[str]
    default: str

