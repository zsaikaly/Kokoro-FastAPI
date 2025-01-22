"""Text processing pipeline."""

from .chunker import split_text
from .normalizer import normalize_text
from .phonemizer import phonemize
from .vocabulary import tokenize


def process_text(text: str, language: str = "a") -> list[int]:
    """Process text through the full pipeline.
    
    Args:
        text: Input text
        language: Language code ('a' for US English, 'b' for British English)
        
    Returns:
        List of token IDs
        
    Note:
        The pipeline:
        1. Converts text to phonemes using phonemizer
        2. Converts phonemes to token IDs using vocabulary
    """
    # Convert text to phonemes
    phonemes = phonemize(text, language=language)
    
    # Convert phonemes to token IDs
    return tokenize(phonemes)
