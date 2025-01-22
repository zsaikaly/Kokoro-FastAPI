from typing import List

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Response
from loguru import logger

from ..services.audio import AudioService
from ..services.text_processing import phonemize, tokenize
from ..services.tts_service import TTSService
from ..structures.text_schemas import (
    GenerateFromPhonemesRequest,
    PhonemeRequest,
    PhonemeResponse,
)

router = APIRouter(tags=["text processing"])


def get_tts_service() -> TTSService:
    """Dependency to get TTSService instance"""
    return TTSService()


@router.post("/text/phonemize", response_model=PhonemeResponse, tags=["deprecated"])
@router.post("/dev/phonemize", response_model=PhonemeResponse)
async def phonemize_text(request: PhonemeRequest) -> PhonemeResponse:
    """Convert text to phonemes and tokens

    Args:
        request: Request containing text and language
        tts_service: Injected TTSService instance

    Returns:
        Phonemes and token IDs
    """
    try:
        if not request.text:
            raise ValueError("Text cannot be empty")

        # Get phonemes
        phonemes = phonemize(request.text, request.language)
        if not phonemes:
            raise ValueError("Failed to generate phonemes")

        # Get tokens
        tokens = tokenize(phonemes)
        tokens = [0] + tokens + [0]  # Add start/end tokens

        return PhonemeResponse(phonemes=phonemes, tokens=tokens)
    except ValueError as e:
        logger.error(f"Error in phoneme generation: {str(e)}")
        raise HTTPException(
            status_code=500, detail={"error": "Server error", "message": str(e)}
        )
    except Exception as e:
        logger.error(f"Error in phoneme generation: {str(e)}")
        raise HTTPException(
            status_code=500, detail={"error": "Server error", "message": str(e)}
        )


@router.post("/text/generate_from_phonemes", tags=["deprecated"])
@router.post("/dev/generate_from_phonemes")
async def generate_from_phonemes(
    request: GenerateFromPhonemesRequest,
    tts_service: TTSService = Depends(get_tts_service),
) -> Response:
    """Generate audio directly from phonemes

    Args:
        request: Request containing phonemes and generation parameters
        tts_service: Injected TTSService instance

    Returns:
        WAV audio bytes
    """
    # Validate phonemes first
    if not request.phonemes:
        raise HTTPException(
            status_code=400,
            detail={"error": "Invalid request", "message": "Phonemes cannot be empty"},
        )

    try:
        # Ensure service is initialized
        await tts_service.ensure_initialized()

        # Validate voice exists
        available_voices = await tts_service.list_voices()
        if request.voice not in available_voices:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Invalid request",
                    "message": f"Voice not found: {request.voice}",
                },
            )

        # Convert phonemes to tokens
        tokens = tokenize(request.phonemes)
        tokens = [0] + tokens + [0]  # Add start/end tokens

        # Generate audio directly from tokens
        audio = await tts_service.model_manager.generate(
            tokens,
            request.voice,
            speed=request.speed
        )

        if audio is None:
            raise ValueError("Failed to generate audio")

        # Convert to WAV bytes
        wav_bytes = AudioService.convert_audio(
            audio, 24000, "wav", is_first_chunk=True, is_last_chunk=True, stream=False
        )

        return Response(
            content=wav_bytes,
            media_type="audio/wav",
            headers={
                "Content-Disposition": "attachment; filename=speech.wav",
                "Cache-Control": "no-cache",
            },
        )

    except ValueError as e:
        logger.error(f"Invalid request: {str(e)}")
        raise HTTPException(
            status_code=400, detail={"error": "Invalid request", "message": str(e)}
        )
    except Exception as e:
        logger.error(f"Error generating audio: {str(e)}")
        raise HTTPException(
            status_code=500, detail={"error": "Server error", "message": str(e)}
        )
