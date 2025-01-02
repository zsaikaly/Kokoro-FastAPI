from typing import List

from loguru import logger
from fastapi import Depends, Response, APIRouter, HTTPException

from ..services.tts import TTSService
from ..services.audio import AudioService
from ..structures.schemas import OpenAISpeechRequest

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
        # Validate voice exists
        available_voices = tts_service.list_voices()
        if request.voice not in available_voices:
            raise ValueError(
                f"Voice '{request.voice}' not found. Available voices: {', '.join(sorted(available_voices))}"
            )

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

    except ValueError as e:
        logger.error(f"Invalid request: {str(e)}")
        raise HTTPException(
            status_code=400, detail={"error": "Invalid request", "message": str(e)}
        )
    except Exception as e:
        logger.error(f"Error generating speech: {str(e)}")
        raise HTTPException(
            status_code=500, detail={"error": "Server error", "message": str(e)}
        )


@router.get("/audio/voices")
async def list_voices(tts_service: TTSService = Depends(get_tts_service)):
    """List all available voices for text-to-speech"""
    try:
        voices = tts_service.list_voices()
        return {"voices": voices}
    except Exception as e:
        logger.error(f"Error listing voices: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/audio/voices/combine")
async def combine_voices(
    request: List[str], tts_service: TTSService = Depends(get_tts_service)
):
    """Combine multiple voices into a new voice.

    Args:
        request: List of voice names to combine

    Returns:
        Dict with combined voice name and list of all available voices

    Raises:
        HTTPException:
            - 400: Invalid request (wrong number of voices, voice not found)
            - 500: Server error (file system issues, combination failed)
    """
    try:
        combined_voice = tts_service.combine_voices(voices=request)
        voices = tts_service.list_voices()
        return {"voices": voices, "voice": combined_voice}

    except ValueError as e:
        logger.error(f"Invalid voice combination request: {str(e)}")
        raise HTTPException(
            status_code=400, detail={"error": "Invalid request", "message": str(e)}
        )

    except RuntimeError as e:
        logger.error(f"Server error during voice combination: {str(e)}")
        raise HTTPException(
            status_code=500, detail={"error": "Server error", "message": str(e)}
        )

    except Exception as e:
        logger.error(f"Unexpected error during voice combination: {str(e)}")
        raise HTTPException(
            status_code=500, detail={"error": "Unexpected error", "message": str(e)}
        )
