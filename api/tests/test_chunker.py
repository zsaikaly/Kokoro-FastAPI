"""Tests for text chunking service"""

import pytest
from api.src.services.text_processing import chunker


def test_split_text():
    """Test text splitting into sentences"""
    text = "First sentence. Second sentence! Third sentence?"
    sentences = list(chunker.split_text(text))
    assert len(sentences) == 3
    assert sentences[0] == "First sentence."
    assert sentences[1] == "Second sentence!"
    assert sentences[2] == "Third sentence?"


def test_split_text_empty():
    """Test splitting empty text"""
    assert list(chunker.split_text("")) == []


def test_split_text_single_sentence():
    """Test splitting single sentence"""
    text = "Just one sentence."
    assert list(chunker.split_text(text)) == ["Just one sentence."]


def test_split_text_with_custom_chunk_size():
    """Test splitting with custom max chunk size"""
    text = "First part, second part, third part."
    chunks = list(chunker.split_text(text, max_chunk=15))
    assert len(chunks) == 3
    assert chunks[0] == "First part,"
    assert chunks[1] == "second part,"
    assert chunks[2] == "third part."
