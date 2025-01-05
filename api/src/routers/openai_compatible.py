from typing import List

from loguru import logger
from fastapi import Depends, Response, APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from ..services.tts_service import TTSService
from ..services.audio import AudioService
from ..structures.schemas import OpenAISpeechRequest
from typing import AsyncGenerator

router = APIRouter(
    tags=["OpenAI Compatible TTS"],
    responses={404: {"description": "Not found"}},
)


def get_tts_service() -> TTSService:
    """Dependency to get TTSService instance with database session"""
    return TTSService()  # Initialize TTSService with default settings


async def stream_audio_chunks(tts_service: TTSService, request: OpenAISpeechRequest) -> AsyncGenerator[bytes, None]:
    """Stream audio chunks as they're generated"""
    async for chunk in tts_service.generate_audio_stream(
        text=request.input,
        voice=request.voice,
        speed=request.speed,
        output_format=request.response_format
    ):
        yield chunk

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

        # Set content type based on format
        content_type = {
            "mp3": "audio/mpeg",
            "opus": "audio/opus",
            "aac": "audio/aac",
            "flac": "audio/flac",
            "wav": "audio/wav",
            "pcm": "audio/pcm",
        }.get(request.response_format, f"audio/{request.response_format}")

        if request.stream:
            # Stream audio chunks as they're generated
            return StreamingResponse(
                stream_audio_chunks(tts_service, request),
                media_type=content_type,
                headers={
                    "Content-Disposition": f"attachment; filename=speech.{request.response_format}",
                    "X-Accel-Buffering": "no",  # Disable proxy buffering
                    "Cache-Control": "no-cache",  # Prevent caching
                },
            )
        else:
            # Generate complete audio
            audio, _ = tts_service._generate_audio(
                text=request.input,
                voice=request.voice,
                speed=request.speed,
                stitch_long_output=True,
            )

            # Convert to requested format
            content = AudioService.convert_audio(
                audio, 
                24000, 
                request.response_format,
                is_first_chunk=True
            )

            return Response(
                content=content,
                media_type=content_type,
                headers={
                    "Content-Disposition": f"attachment; filename=speech.{request.response_format}",
                    "Cache-Control": "no-cache",  # Prevent caching
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
