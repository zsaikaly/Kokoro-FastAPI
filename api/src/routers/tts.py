import os
from fastapi import APIRouter, HTTPException, Response, Depends
from sqlalchemy.orm import Session
from ..models.schemas import TTSRequest, TTSResponse, VoicesResponse
from ..services.tts import TTSService
from ..database.database import get_db

router = APIRouter(
    prefix="/tts",
    tags=["TTS"],
    responses={404: {"description": "Not found"}},
)

def get_tts_service(db: Session = Depends(get_db)) -> TTSService:
    """Dependency to get TTSService instance with database session"""
    return TTSService(db)


@router.get("/voices", response_model=VoicesResponse)
async def get_voices(tts_service: TTSService = Depends(get_tts_service)):
    """List all available voices"""
    voices = tts_service.list_voices()
    return {"voices": voices, "default": "af"}


@router.post("", response_model=TTSResponse)
async def create_tts(request: TTSRequest, tts_service: TTSService = Depends(get_tts_service)):
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
        request.speed,
        request.stitch_long_output
    )
    return {
        "request_id": request_id,
        "status": "pending",
        "output_file": None,
        "processing_time": None,
    }


@router.get("/{request_id}", response_model=TTSResponse)
async def get_status(request_id: int, tts_service: TTSService = Depends(get_tts_service)):
    """Check the status of a TTS request"""
    request = tts_service.get_request_status(request_id)
    if not request:
        raise HTTPException(status_code=404, detail="Request not found")

    return {
        "request_id": request.id,
        "status": request.status,
        "output_file": request.output_file,
        "processing_time": request.processing_time,
    }


@router.get("/file/{request_id}")
async def get_file(request_id: int, tts_service: TTSService = Depends(get_tts_service)):
    """Download the generated audio file"""
    request = tts_service.get_request_status(request_id)
    if not request:
        raise HTTPException(status_code=404, detail="Request not found")

    if request.status != "completed":
        raise HTTPException(status_code=400, detail="Audio generation not complete")

    if not request.output_file or not os.path.exists(request.output_file):
        raise HTTPException(status_code=404, detail="Audio file not found")

    # Read file and ensure it's closed after
    with open(request.output_file, "rb") as f:
        content = f.read()

    return Response(
        content=content,
        media_type="audio/wav",
        headers={
            "Content-Disposition": f"attachment; filename=speech_{request_id}.wav"
        },
    )
