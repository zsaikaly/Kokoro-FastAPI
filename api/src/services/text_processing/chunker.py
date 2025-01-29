# """Text chunking module for TTS processing"""

# from typing import List, AsyncGenerator

# async def fallback_split(text: str, max_chars: int = 400) -> List[str]:
#     """Emergency length control - only used if chunks are too long"""
#     words = text.split()
#     chunks = []
#     current = []
#     current_len = 0
    
#     for word in words:
#         # Always include at least one word per chunk
#         if not current:
#             current.append(word)
#             current_len = len(word)
#             continue
            
#         # Check if adding word would exceed limit
#         if current_len + len(word) + 1 <= max_chars:
#             current.append(word)
#             current_len += len(word) + 1
#         else:
#             chunks.append(" ".join(current))
#             current = [word]
#             current_len = len(word)
    
#     if current:
#         chunks.append(" ".join(current))
    
#     return chunks
