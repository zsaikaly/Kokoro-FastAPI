import os
from fastapi import APIRouter, HTTPException, Response
from ..models.schemas import TTSRequest, TTSResponse, VoicesResponse
from ..services.tts import TTSService

router = APIRouter(
    prefix="/tts",
    tags=["TTS"],
    responses={404: {"description": "Not found"}},
)

# Initialize TTS service
tts_service = TTSService()


@router.get("/voices", response_model=VoicesResponse)
async def get_voices():
    """List all available voices"""
    voices = tts_service.list_voices()
    return {"voices": voices, "default": "af"}


@router.post("", response_model=TTSResponse)
async def create_tts(request: TTSRequest):
    """Submit text for TTS generation"""
    # Validate voice exists
    voices = tts_service.list_voices()
    if request.voice not in voices:
        raise HTTPException(
            status_code=400,
            detail=f"Voice '{request.voice}' not found. Available voices: {voices}",
        )

    # Queue the request
    request_id = tts_service.create_tts_request(
        request.text, 
        request.voice,
        request.stitch_long_output
    )
    return {
        "request_id": request_id,
        "status": "pending",
        "output_file": None,
        "processing_time": None,
    }


@router.get("/{request_id}", response_model=TTSResponse)
async def get_status(request_id: int):
    """Check the status of a TTS request"""
    status = tts_service.get_request_status(request_id)
    if not status:
        raise HTTPException(status_code=404, detail="Request not found")

    status_str, output_file, processing_time = status
    return {
        "request_id": request_id,
        "status": status_str,
        "output_file": output_file,
        "processing_time": processing_time,
    }


@router.get("/file/{request_id}")
async def get_file(request_id: int):
    """Download the generated audio file"""
    status = tts_service.get_request_status(request_id)
    if not status:
        raise HTTPException(status_code=404, detail="Request not found")

    status_str, output_file, _ = status
    if status_str != "completed":
        raise HTTPException(status_code=400, detail="Audio generation not complete")

    if not output_file or not os.path.exists(output_file):
        raise HTTPException(status_code=404, detail="Audio file not found")

    # Read file and ensure it's closed after
    with open(output_file, "rb") as f:
        content = f.read()

    return Response(
        content=content,
        media_type="audio/wav",
        headers={
            "Content-Disposition": f"attachment; filename=speech_{request_id}.wav"
        },
    )
