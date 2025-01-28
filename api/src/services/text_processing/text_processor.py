"""Unified text processing for TTS with smart chunking."""

import re
import time
from typing import AsyncGenerator, List, Tuple
from loguru import logger
from .phonemizer import phonemize
from .normalizer import normalize_text
from .vocabulary import tokenize

def process_text_chunk(text: str, language: str = "a") -> List[int]:
    """Process a chunk of text through normalization, phonemization, and tokenization.
    
    Args:
        text: Text chunk to process
        language: Language code for phonemization
        
    Returns:
        List of token IDs
    """
    start_time = time.time()
    
    # Normalize
    t0 = time.time()
    normalized = normalize_text(text)
    t1 = time.time()
    logger.debug(f"Normalization took {(t1-t0)*1000:.2f}ms for {len(text)} chars")
    
    # Phonemize
    t0 = time.time()
    phonemes = phonemize(normalized, language, normalize=False)  # Already normalized
    t1 = time.time()
    logger.debug(f"Phonemization took {(t1-t0)*1000:.2f}ms for {len(normalized)} chars")
    
    # Convert to token IDs
    t0 = time.time()
    tokens = tokenize(phonemes)
    t1 = time.time()
    logger.debug(f"Tokenization took {(t1-t0)*1000:.2f}ms for {len(phonemes)} chars")
    
    total_time = time.time() - start_time
    logger.debug(f"Total processing took {total_time*1000:.2f}ms for chunk: '{text[:50]}...'")
    
    return tokens

def process_text(text: str, language: str = "a") -> List[int]:
    """Process text into token IDs.
    
    Args:
        text: Text to process
        language: Language code for phonemization
        
    Returns:
        List of token IDs
    """
    if not isinstance(text, str):
        text = str(text) if text is not None else ""
        
    text = text.strip()
    if not text:
        return []
        
    return process_text_chunk(text, language)

async def smart_split(text: str, max_tokens: int = 500) -> AsyncGenerator[Tuple[str, List[int]], None]:
    """Split text into semantically meaningful chunks while respecting token limits.
    
    Args:
        text: Input text to split
        max_tokens: Maximum tokens per chunk
        
    Yields:
        Tuples of (text chunk, token IDs) where token count is <= max_tokens
    """
    start_time = time.time()
    chunk_count = 0
    total_chars = len(text)
    logger.info(f"Starting text split for {total_chars} characters with {max_tokens} max tokens")
    
    # Split on major punctuation first
    sentences = re.split(r'([.!?;:])', text)
    
    current_chunk = []
    current_token_count = 0
    
    for i in range(0, len(sentences), 2):
        # Get sentence and its punctuation (if any)
        sentence = sentences[i].strip()
        punct = sentences[i + 1] if i + 1 < len(sentences) else ""
        
        if not sentence:
            continue
            
        # Process sentence to get token count
        sentence_with_punct = sentence + punct
        tokens = process_text_chunk(sentence_with_punct)
        token_count = len(tokens)
        logger.debug(f"Sentence '{sentence_with_punct[:50]}...' has {token_count} tokens")
        
        # If this single sentence is too long, split on commas
        if token_count > max_tokens:
            logger.debug(f"Sentence exceeds token limit, splitting on commas")
            clause_splits = re.split(r'([,])', sentence_with_punct)
            for j in range(0, len(clause_splits), 2):
                clause = clause_splits[j].strip()
                comma = clause_splits[j + 1] if j + 1 < len(clause_splits) else ""
                
                if not clause:
                    continue
                    
                clause_with_punct = clause + comma
                clause_tokens = process_text_chunk(clause_with_punct)
                
                # If still too long, do a hard split on words
                if len(clause_tokens) > max_tokens:
                    logger.debug(f"Clause exceeds token limit, splitting on words")
                    words = clause_with_punct.split()
                    temp_chunk = []
                    temp_tokens = []
                    
                    for word in words:
                        word_tokens = process_text_chunk(word)
                        if len(temp_tokens) + len(word_tokens) > max_tokens:
                            if temp_chunk:  # Don't yield empty chunks
                                chunk_text = " ".join(temp_chunk)
                                chunk_count += 1
                                logger.info(f"Yielding word-split chunk {chunk_count}: '{chunk_text[:50]}...' ({len(temp_tokens)} tokens)")
                                yield chunk_text, temp_tokens
                            temp_chunk = [word]
                            temp_tokens = word_tokens
                        else:
                            temp_chunk.append(word)
                            temp_tokens.extend(word_tokens)
                    
                    if temp_chunk:  # Don't forget the last chunk
                        chunk_text = " ".join(temp_chunk)
                        chunk_count += 1
                        logger.info(f"Yielding final word-split chunk {chunk_count}: '{chunk_text[:50]}...' ({len(temp_tokens)} tokens)")
                        yield chunk_text, temp_tokens
                        
                else:
                    # Check if adding this clause would exceed the limit
                    if current_token_count + len(clause_tokens) > max_tokens:
                        if current_chunk:  # Don't yield empty chunks
                            chunk_text = " ".join(current_chunk)
                            chunk_count += 1
                            logger.info(f"Yielding clause-split chunk {chunk_count}: '{chunk_text[:50]}...' ({current_token_count} tokens)")
                            yield chunk_text, process_text_chunk(chunk_text)
                        current_chunk = [clause_with_punct]
                        current_token_count = len(clause_tokens)
                    else:
                        current_chunk.append(clause_with_punct)
                        current_token_count += len(clause_tokens)
        
        else:
            # Check if adding this sentence would exceed the limit
            if current_token_count + token_count > max_tokens:
                if current_chunk:  # Don't yield empty chunks
                    chunk_text = " ".join(current_chunk)
                    chunk_count += 1
                    logger.info(f"Yielding sentence-split chunk {chunk_count}: '{chunk_text[:50]}...' ({current_token_count} tokens)")
                    yield chunk_text, process_text_chunk(chunk_text)
                current_chunk = [sentence_with_punct]
                current_token_count = token_count
            else:
                current_chunk.append(sentence_with_punct)
                current_token_count += token_count
    
    # Don't forget the last chunk
    if current_chunk:
        chunk_text = " ".join(current_chunk)
        chunk_count += 1
        logger.info(f"Yielding final chunk {chunk_count}: '{chunk_text[:50]}...' ({current_token_count} tokens)")
        yield chunk_text, process_text_chunk(chunk_text)
    
    total_time = time.time() - start_time
    logger.info(f"Text splitting completed in {total_time*1000:.2f}ms, produced {chunk_count} chunks")
