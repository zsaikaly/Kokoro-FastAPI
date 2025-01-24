from __future__ import annotations
import re
from typing import Callable

# Prioritize sentence boundaries for TTS
_NON_WHITESPACE_SEMANTIC_SPLITTERS = (
    '.', '!', '?',  # Primary - sentence boundaries
    ';', ':',       # Secondary - major clause boundaries
    ',',           # Tertiary - minor clause boundaries
    '(', ')', '[', ']', '"', '"', "'", "'", "'", '"', '`',  # Other punctuation
    '—', '…',      # Dashes and ellipsis
    '/', '\\', '–', '&', '-',  # Word joiners
)
"""Semantic splitters ordered by priority for TTS chunking"""

def _split_text(text: str) -> tuple[str, bool, list[str]]:
    """Split text using the most semantically meaningful splitter possible."""
    
    splitter_is_whitespace = True

    # Try splitting at, in order:
    # - Newlines (natural paragraph breaks)
    # - Spaces (if no other splits possible)
    # - Semantic splitters (prioritizing sentence boundaries)
    if '\n' in text or '\r' in text:
        splitter = max(re.findall(r'[\r\n]+', text))
    
    elif re.search(r'\s', text):
        splitter = max(re.findall(r'\s+', text))
    
    else:
        # Find first semantic splitter present
        for splitter in _NON_WHITESPACE_SEMANTIC_SPLITTERS:
            if splitter in text:
                splitter_is_whitespace = False
                break
        else:
            return '', splitter_is_whitespace, list(text)
    
    return splitter, splitter_is_whitespace, text.split(splitter)

class Chunker:
    def __init__(self, chunk_size: int, token_counter: Callable[[str], int]) -> None:
        self.chunk_size = chunk_size
        self.token_counter = token_counter
    
    def __call__(self, text: str) -> list[str]:
        """Split text into chunks based on semantic boundaries."""
        if not isinstance(text, str):
            text = str(text) if text is not None else ""
            
        text = text.strip()
        if not text:
            return []
            
        # Split the text
        splitter, _, splits = _split_text(text)
        
        chunks = []
        current_chunk = []
        current_len = 0
        
        for split in splits:
            split = split.strip()
            if not split:
                continue
                
            # Check if adding this split would exceed chunk size
            split_len = self.token_counter(split)
            if current_len + split_len <= self.chunk_size:
                current_chunk.append(split)
                current_len += split_len
            else:
                # Save current chunk if it exists
                if current_chunk:
                    chunks.append(splitter.join(current_chunk))
                # Start new chunk with current split
                current_chunk = [split]
                current_len = split_len
        
        # Add final chunk if it exists
        if current_chunk:
            chunks.append(splitter.join(current_chunk))
            
        return chunks

def chunkerify(token_counter: Callable[[str], int], chunk_size: int) -> Chunker:
    """Create a chunker with the specified token counter and chunk size."""
    return Chunker(chunk_size=chunk_size, token_counter=token_counter)