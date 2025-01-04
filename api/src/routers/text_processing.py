from fastapi import APIRouter
from ..structures.text_schemas import PhonemeRequest, PhonemeResponse
from ..services.text_processing import phonemize, tokenize

router = APIRouter(
    prefix="/text",
    tags=["text processing"]
)

@router.post("/phonemize", response_model=PhonemeResponse)
async def phonemize_text(request: PhonemeRequest) -> PhonemeResponse:
    """Convert text to phonemes and tokens: Rough attempt
    
    Args:
        request: Request containing text and language
        
    Returns:
        Phonemes and token IDs
    """
    # Get phonemes
    phonemes = phonemize(request.text, request.language)
    
    # Get tokens
    tokens = tokenize(phonemes)
    tokens = [0] + tokens + [0]  # Add start/end tokens
    
    return PhonemeResponse(
        phonemes=phonemes,
        tokens=tokens
    )
