import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from fastapi.testclient import TestClient
import numpy as np
import asyncio
from typing import AsyncGenerator

from api.src.main import app
from api.src.services.tts_service import TTSService

client = TestClient(app)

@pytest.fixture
def mock_audio_bytes():
    """Mock audio bytes for testing."""
    return b"mock audio data"

@pytest.fixture
def mock_tts_service(mock_audio_bytes):
    """Mock TTS service for testing."""
    with patch("api.src.routers.openai_compatible.get_tts_service") as mock_get:
        service = AsyncMock(spec=TTSService)
        service.generate_audio.return_value = (np.zeros(1000), 0.1)
        
        # Create a proper async generator for streaming
        async def mock_stream(*args, **kwargs) -> AsyncGenerator[bytes, None]:
            yield mock_audio_bytes
        
        service.generate_audio_stream = mock_stream
        service.list_voices.return_value = ["voice1", "voice2"]
        service.combine_voices.return_value = "voice1_voice2"
        
        # Return the same instance for all calls
        mock_get.return_value = service
        mock_get.side_effect = None
        yield service

def test_openai_speech_endpoint(mock_tts_service, test_voice):
    """Test the OpenAI-compatible speech endpoint with basic MP3 generation"""
    response = client.post(
        "/v1/audio/speech",
        json={
            "model": "kokoro",
            "input": "Hello world",
            "voice": test_voice,
            "response_format": "mp3",
            "stream": False
        }
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/mpeg"
    assert len(response.content) > 0

def test_openai_speech_streaming(mock_tts_service, test_voice, mock_audio_bytes):
    """Test the OpenAI-compatible speech endpoint with streaming"""
    response = client.post(
        "/v1/audio/speech",
        json={
            "model": "kokoro",
            "input": "Hello world",
            "voice": test_voice,
            "response_format": "mp3",
            "stream": True
        }
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/mpeg"
    assert "Transfer-Encoding" in response.headers
    assert response.headers["Transfer-Encoding"] == "chunked"
    
    # For streaming responses, we need to read the content in chunks
    content = b""
    for chunk in response.iter_bytes():
        content += chunk
    assert content == mock_audio_bytes

def test_openai_speech_pcm_streaming(mock_tts_service, test_voice, mock_audio_bytes):
    """Test PCM streaming format"""
    response = client.post(
        "/v1/audio/speech",
        json={
            "model": "kokoro",
            "input": "Hello world",
            "voice": test_voice,
            "response_format": "pcm",
            "stream": True
        }
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/pcm"
    
    # For streaming responses, we need to read the content in chunks
    content = b""
    for chunk in response.iter_bytes():
        content += chunk
    assert content == mock_audio_bytes

def test_openai_speech_invalid_voice(mock_tts_service):
    """Test error handling for invalid voice"""
    mock_tts_service.generate_audio.side_effect = ValueError("Voice 'invalid_voice' not found")
    
    response = client.post(
        "/v1/audio/speech",
        json={
            "model": "kokoro",
            "input": "Hello world",
            "voice": "invalid_voice",
            "response_format": "mp3",
            "stream": False
        }
    )
    assert response.status_code == 400
    error_response = response.json()
    assert error_response["detail"]["error"] == "validation_error"
    assert "Voice 'invalid_voice' not found" in error_response["detail"]["message"]
    assert error_response["detail"]["type"] == "invalid_request_error"

def test_openai_speech_empty_text(mock_tts_service, test_voice):
    """Test error handling for empty text"""
    mock_tts_service.generate_audio.side_effect = ValueError("Text is empty after preprocessing")
    
    response = client.post(
        "/v1/audio/speech",
        json={
            "model": "kokoro",
            "input": "",
            "voice": test_voice,
            "response_format": "mp3",
            "stream": False
        }
    )
    assert response.status_code == 400
    error_response = response.json()
    assert error_response["detail"]["error"] == "validation_error"
    assert "Text is empty after preprocessing" in error_response["detail"]["message"]
    assert error_response["detail"]["type"] == "invalid_request_error"

def test_openai_speech_invalid_format(mock_tts_service, test_voice):
    """Test error handling for invalid format"""
    response = client.post(
        "/v1/audio/speech",
        json={
            "model": "kokoro",
            "input": "Hello world",
            "voice": test_voice,
            "response_format": "invalid_format",
            "stream": False
        }
    )
    assert response.status_code == 422  # Validation error from Pydantic

def test_list_voices(mock_tts_service):
    """Test listing available voices"""
    response = client.get("/v1/audio/voices")
    assert response.status_code == 200
    data = response.json()
    assert "voices" in data
    assert len(data["voices"]) == 2
    assert "voice1" in data["voices"]
    assert "voice2" in data["voices"]

def test_combine_voices(mock_tts_service):
    """Test combining voices endpoint"""
    response = client.post(
        "/v1/audio/voices/combine",
        json="voice1+voice2"
    )
    assert response.status_code == 200
    data = response.json()
    assert "voice" in data
    assert data["voice"] == "voice1_voice2"

def test_server_error(mock_tts_service, test_voice):
    """Test handling of server errors"""
    mock_tts_service.generate_audio.side_effect = RuntimeError("Internal server error")
    
    response = client.post(
        "/v1/audio/speech",
        json={
            "model": "kokoro",
            "input": "Hello world",
            "voice": test_voice,
            "response_format": "mp3",
            "stream": False
        }
    )
    assert response.status_code == 500
    error_response = response.json()
    assert error_response["detail"]["error"] == "processing_error"
    assert error_response["detail"]["type"] == "server_error"

def test_streaming_error(mock_tts_service, test_voice):
    """Test handling streaming errors"""
    # Create a proper async generator that raises an error
    async def mock_error_stream(*args, **kwargs) -> AsyncGenerator[bytes, None]:
        if False:  # This makes it a proper generator
            yield b""
        raise RuntimeError("Streaming failed")
    
    mock_tts_service.generate_audio_stream = mock_error_stream
    
    response = client.post(
        "/v1/audio/speech",
        json={
            "model": "kokoro",
            "input": "Hello world",
            "voice": test_voice,
            "response_format": "mp3",
            "stream": True
        }
    )
    assert response.status_code == 500
    error_response = response.json()
    assert error_response["detail"]["error"] == "processing_error"
    assert error_response["detail"]["type"] == "server_error"