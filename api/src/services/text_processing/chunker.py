"""Text chunking module for TTS processing"""

from typing import List, AsyncGenerator
from . import semchunk_slim

async def fallback_split(text: str, max_chars: int = 400) -> List[str]:
    """Emergency length control - only used if chunks are too long"""
    words = text.split()
    chunks = []
    current = []
    current_len = 0
    
    for word in words:
        # Always include at least one word per chunk
        if not current:
            current.append(word)
            current_len = len(word)
            continue
            
        # Check if adding word would exceed limit
        if current_len + len(word) + 1 <= max_chars:
            current.append(word)
            current_len += len(word) + 1
        else:
            chunks.append(" ".join(current))
            current = [word]
            current_len = len(word)
    
    if current:
        chunks.append(" ".join(current))
    
    return chunks

async def split_text(text: str, max_chunk: int = None) -> AsyncGenerator[str, None]:
    """Split text into TTS-friendly chunks
    
    Args:
        text: Text to split into chunks
        max_chunk: Maximum chunk size (defaults to 400)
        
    Yields:
        Text chunks suitable for TTS processing
    """
    if max_chunk is None:
        max_chunk = 400
        
    if not isinstance(text, str):
        text = str(text) if text is not None else ""
        
    text = text.strip()
    if not text:
        return
        
    # Initialize chunker targeting ~300 chars to allow for expansion
    chunker = semchunk_slim.chunkerify(
        lambda t: len(t) // 5,  # Simple length-based target
        chunk_size=60  # Target ~300 chars
    )
    
    # Get initial chunks
    chunks = chunker(text)
    
    # Process chunks
    for chunk in chunks:
        chunk = chunk.strip()
        if not chunk:
            continue
            
        # Use fallback for any chunks that are too long
        if len(chunk) > max_chunk:
            for subchunk in await fallback_split(chunk, max_chunk):
                yield subchunk
        else:
            yield chunk
