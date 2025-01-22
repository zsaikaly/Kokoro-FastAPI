from typing import AsyncGenerator, List, Union

from fastapi import APIRouter, Depends, Header, HTTPException, Request, Response
from fastapi.responses import StreamingResponse
from loguru import logger

from ..services.audio import AudioService
from ..services.tts_service import TTSService
from ..structures.schemas import OpenAISpeechRequest

router = APIRouter(
    tags=["OpenAI Compatible TTS"],
    responses={404: {"description": "Not found"}},
)

# Global TTSService instance with lock
_tts_service = None
_init_lock = None

async def get_tts_service() -> TTSService:
    """Get global TTSService instance"""
    global _tts_service, _init_lock
    
    # Create lock if needed
    if _init_lock is None:
        import asyncio
        _init_lock = asyncio.Lock()
    
    # Initialize service if needed
    if _tts_service is None:
        async with _init_lock:
            # Double check pattern
            if _tts_service is None:
                _tts_service = await TTSService.create()
                logger.info("Created global TTSService instance")
    
    return _tts_service


async def process_voices(
    voice_input: Union[str, List[str]], tts_service: TTSService
) -> str:
    """Process voice input into a combined voice, handling both string and list formats"""
    # Convert input to list of voices
    if isinstance(voice_input, str):
        voices = [v.strip() for v in voice_input.split("+") if v.strip()]
    else:
        voices = voice_input

    if not voices:
        raise ValueError("No voices provided")

    # Check if all voices exist
    available_voices = await tts_service.list_voices()
    for voice in voices:
        if voice not in available_voices:
            raise ValueError(
                f"Voice '{voice}' not found. Available voices: {', '.join(sorted(available_voices))}"
            )

    # If single voice, return it directly
    if len(voices) == 1:
        return voices[0]

    # Otherwise combine voices
    return await tts_service.combine_voices(voices=voices)


async def stream_audio_chunks(
    tts_service: TTSService, 
    request: OpenAISpeechRequest,
    client_request: Request
) -> AsyncGenerator[bytes, None]:
    """Stream audio chunks as they're generated with client disconnect handling"""
    voice_to_use = await process_voices(request.voice, tts_service)
    
    try:
        async for chunk in tts_service.generate_audio_stream(
            text=request.input,
            voice=voice_to_use,
            speed=request.speed,
            output_format=request.response_format,
        ):
            # Check if client is still connected
            if await client_request.is_disconnected():
                logger.info("Client disconnected, stopping audio generation")
                break
            yield chunk
    except Exception as e:
        logger.error(f"Error in audio streaming: {str(e)}")
        # Let the exception propagate to trigger cleanup
        raise


@router.post("/audio/speech")
async def create_speech(
    request: OpenAISpeechRequest,
    client_request: Request,
    x_raw_response: str = Header(None, alias="x-raw-response"),
):
    """OpenAI-compatible endpoint for text-to-speech"""
    try:
        # Get global service instance
        tts_service = await get_tts_service()
        
        # Process voice combination and validate
        voice_to_use = await process_voices(request.voice, tts_service)

        # Set content type based on format
        content_type = {
            "mp3": "audio/mpeg",
            "opus": "audio/opus",
            "aac": "audio/aac",
            "flac": "audio/flac",
            "wav": "audio/wav",
            "pcm": "audio/pcm",
        }.get(request.response_format, f"audio/{request.response_format}")

        # Check if streaming is requested (default for OpenAI client)
        if request.stream:
            # Stream audio chunks as they're generated
            return StreamingResponse(
                stream_audio_chunks(tts_service, request, client_request),
                media_type=content_type,
                headers={
                    "Content-Disposition": f"attachment; filename=speech.{request.response_format}",
                    "X-Accel-Buffering": "no",  # Disable proxy buffering
                    "Cache-Control": "no-cache",  # Prevent caching
                    "Transfer-Encoding": "chunked",  # Enable chunked transfer encoding
                },
            )
        else:
            # Generate complete audio using public interface
            audio, _ = await tts_service.generate_audio(
                text=request.input,
                voice=voice_to_use,
                speed=request.speed,
                stitch_long_output=True
            )

            # Convert to requested format
            content = AudioService.convert_audio(
                audio, 24000, request.response_format, is_first_chunk=True, stream=False
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
        # Handle validation errors
        logger.warning(f"Invalid request: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail={
                "error": "validation_error",
                "message": str(e),
                "type": "invalid_request_error"
            }
        )
    except RuntimeError as e:
        # Handle runtime/processing errors
        logger.error(f"Processing error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "processing_error",
                "message": "Failed to process audio generation request",
                "type": "server_error"
            }
        )
    except Exception as e:
        # Handle unexpected errors
        logger.error(f"Unexpected error in speech generation: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "server_error",
                "message": "An unexpected error occurred",
                "type": "server_error"
            }
        )


@router.get("/audio/voices")
async def list_voices():
    """List all available voices for text-to-speech"""
    try:
        tts_service = await get_tts_service()
        voices = await tts_service.list_voices()
        return {"voices": voices}
    except Exception as e:
        logger.error(f"Error listing voices: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "server_error",
                "message": "Failed to retrieve voice list",
                "type": "server_error"
            }
        )


@router.post("/audio/voices/combine")
async def combine_voices(request: Union[str, List[str]]):
    """Combine multiple voices into a new voice.

    Args:
        request: Either a string with voices separated by + (e.g. "voice1+voice2")
                or a list of voice names to combine

    Returns:
        Dict with combined voice name and list of all available voices

    Raises:
        HTTPException:
            - 400: Invalid request (wrong number of voices, voice not found)
            - 500: Server error (file system issues, combination failed)
    """
    try:
        tts_service = await get_tts_service()
        combined_voice = await process_voices(request, tts_service)
        voices = await tts_service.list_voices()
        return {"voices": voices, "voice": combined_voice}

    except ValueError as e:
        logger.warning(f"Invalid voice combination request: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail={
                "error": "validation_error",
                "message": str(e),
                "type": "invalid_request_error"
            }
        )
    except RuntimeError as e:
        logger.error(f"Voice combination processing error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "processing_error",
                "message": "Failed to process voice combination request",
                "type": "server_error"
            }
        )
    except Exception as e:
        logger.error(f"Unexpected error in voice combination: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "server_error",
                "message": "An unexpected error occurred",
                "type": "server_error"
            }
        )
