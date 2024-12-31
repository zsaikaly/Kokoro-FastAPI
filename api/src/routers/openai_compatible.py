from fastapi import APIRouter, HTTPException, Response, Depends
import logging
from ..structures.schemas import OpenAISpeechRequest
from ..services.tts import TTSService
from ..services.audio import AudioService

logger = logging.getLogger(__name__)

router = APIRouter(
    tags=["OpenAI Compatible TTS"],
    responses={404: {"description": "Not found"}},
)


def get_tts_service() -> TTSService:
    """Dependency to get TTSService instance with database session"""
    return TTSService(
        start_worker=False
    )  # Don't start worker thread for OpenAI endpoint


@router.post("/audio/speech")
async def create_speech(
    request: OpenAISpeechRequest, tts_service: TTSService = Depends(get_tts_service)
):
    """OpenAI-compatible endpoint for text-to-speech"""
    try:
        # Generate audio directly using TTSService's method
        audio, _ = tts_service._generate_audio(
            text=request.input,
            voice=request.voice,
            speed=request.speed,
            stitch_long_output=True,
        )

        # Convert to requested format
        content = AudioService.convert_audio(audio, 24000, request.response_format)

        return Response(
            content=content,
            media_type=f"audio/{request.response_format}",
            headers={
                "Content-Disposition": f"attachment; filename=speech.{request.response_format}"
            },
        )

    except Exception as e:
        logger.error(f"Error generating speech: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/audio/voices")
async def list_voices(tts_service: TTSService = Depends(get_tts_service)):
    """List all available voices for text-to-speech"""
    try:
        voices = tts_service.list_voices()
        return {"voices": voices}
    except Exception as e:
        logger.error(f"Error listing voices: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
