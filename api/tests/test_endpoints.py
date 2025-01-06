from unittest.mock import Mock, AsyncMock

import pytest
import pytest_asyncio
import asyncio
from fastapi.testclient import TestClient
from httpx import AsyncClient

from ..src.main import app

# Create test client
client = TestClient(app)

# Create async client fixture
@pytest_asyncio.fixture
async def async_client():
    async with AsyncClient(app=app, base_url="http://test") as ac:
        yield ac


# Mock services
@pytest.fixture
def mock_tts_service(monkeypatch):
    mock_service = Mock()
    mock_service._generate_audio.return_value = (bytes([0, 1, 2, 3]), 1.0)
    
    # Create proper async generator mock
    async def mock_stream(*args, **kwargs):
        for chunk in [b"chunk1", b"chunk2"]:
            yield chunk
    mock_service.generate_audio_stream = mock_stream
    mock_service.list_voices.return_value = [
        "af",
        "bm_lewis",
        "bf_isabella",
        "bf_emma",
        "af_sarah",
        "af_bella",
        "am_adam",
        "am_michael",
        "bm_george",
    ]
    monkeypatch.setattr(
        "api.src.routers.openai_compatible.TTSService",
        lambda *args, **kwargs: mock_service,
    )
    return mock_service


@pytest.fixture
def mock_audio_service(monkeypatch):
    mock_service = Mock()
    mock_service.convert_audio.return_value = b"converted mock audio data"
    monkeypatch.setattr(
        "api.src.routers.openai_compatible.AudioService", mock_service
    )
    return mock_service


def test_health_check():
    """Test the health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_openai_speech_endpoint(mock_tts_service, mock_audio_service):
    """Test the OpenAI-compatible speech endpoint"""
    test_request = {
        "model": "kokoro",
        "input": "Hello world",
        "voice": "bm_lewis",
        "response_format": "wav",
        "speed": 1.0,
        "stream": False  # Explicitly disable streaming
    }
    response = client.post("/v1/audio/speech", json=test_request)
    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/wav"
    assert response.headers["content-disposition"] == "attachment; filename=speech.wav"
    mock_tts_service._generate_audio.assert_called_once_with(
        text="Hello world", voice="bm_lewis", speed=1.0, stitch_long_output=True
    )
    assert response.content == b"converted mock audio data"


def test_openai_speech_invalid_voice(mock_tts_service):
    """Test the OpenAI-compatible speech endpoint with invalid voice"""
    test_request = {
        "model": "kokoro",
        "input": "Hello world",
        "voice": "invalid_voice",
        "response_format": "wav",
        "speed": 1.0,
        "stream": False  # Explicitly disable streaming
    }
    response = client.post("/v1/audio/speech", json=test_request)
    assert response.status_code == 400  # Bad request
    assert "not found" in response.json()["detail"]["message"]


def test_openai_speech_invalid_speed(mock_tts_service):
    """Test the OpenAI-compatible speech endpoint with invalid speed"""
    test_request = {
        "model": "kokoro",
        "input": "Hello world",
        "voice": "af",
        "response_format": "wav",
        "speed": -1.0,  # Invalid speed
        "stream": False  # Explicitly disable streaming
    }
    response = client.post("/v1/audio/speech", json=test_request)
    assert response.status_code == 422  # Validation error


def test_openai_speech_generation_error(mock_tts_service):
    """Test error handling in speech generation"""
    mock_tts_service._generate_audio.side_effect = Exception("Generation failed")
    test_request = {
        "model": "kokoro",
        "input": "Hello world",
        "voice": "af",
        "response_format": "wav",
        "speed": 1.0,
        "stream": False  # Explicitly disable streaming
    }
    response = client.post("/v1/audio/speech", json=test_request)
    assert response.status_code == 500
    assert "Generation failed" in response.json()["detail"]["message"]


def test_combine_voices_success(mock_tts_service):
    """Test successful voice combination"""
    test_voices = ["af_bella", "af_sarah"]
    mock_tts_service.combine_voices.return_value = "af_bella_af_sarah"

    response = client.post("/v1/audio/voices/combine", json=test_voices)

    assert response.status_code == 200
    assert response.json()["voice"] == "af_bella_af_sarah"
    mock_tts_service.combine_voices.assert_called_once_with(voices=test_voices)


def test_combine_voices_single_voice(mock_tts_service):
    """Test combining single voice returns default voice"""
    test_voices = ["af_bella"]
    mock_tts_service.combine_voices.return_value = "af"

    response = client.post("/v1/audio/voices/combine", json=test_voices)

    assert response.status_code == 200
    assert response.json()["voice"] == "af"


def test_combine_voices_empty_list(mock_tts_service):
    """Test combining empty voice list returns default voice"""
    test_voices = []
    mock_tts_service.combine_voices.return_value = "af"

    response = client.post("/v1/audio/voices/combine", json=test_voices)

    assert response.status_code == 200
    assert response.json()["voice"] == "af"


def test_combine_voices_error(mock_tts_service):
    """Test error handling in voice combination"""
    test_voices = ["af_bella", "af_sarah"]
    mock_tts_service.combine_voices.side_effect = Exception("Combination failed")

    response = client.post("/v1/audio/voices/combine", json=test_voices)

    assert response.status_code == 500
    assert "Combination failed" in response.json()["detail"]["message"]


@pytest.mark.asyncio
async def test_openai_speech_pcm_streaming(mock_tts_service, async_client):
    """Test streaming PCM audio for real-time playback"""
    test_request = {
        "model": "kokoro",
        "input": "Hello world",
        "voice": "af",
        "response_format": "pcm",
        "stream": True
    }
    
    # Create streaming mock for this test
    async def mock_stream(*args, **kwargs):
        for chunk in [b"chunk1", b"chunk2"]:
            yield chunk
    mock_tts_service.generate_audio_stream = mock_stream
    
    # Add streaming header
    headers = {"x-raw-response": "stream"}
    response = await async_client.post("/v1/audio/speech", json=test_request, headers=headers)
    
    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/pcm"
    # Just verify status and content type
    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/pcm"


@pytest.mark.asyncio
async def test_openai_speech_streaming_mp3(mock_tts_service, async_client):
    """Test streaming MP3 audio to file"""
    test_request = {
        "model": "kokoro",
        "input": "Hello world",
        "voice": "af",
        "response_format": "mp3",
        "stream": True
    }
    
    # Create streaming mock for this test
    async def mock_stream(*args, **kwargs):
        for chunk in [b"mp3header", b"mp3data"]:
            yield chunk
    mock_tts_service.generate_audio_stream = mock_stream
    
    # Add streaming header
    headers = {"x-raw-response": "stream"}
    response = await async_client.post("/v1/audio/speech", json=test_request, headers=headers)
    
    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/mpeg"
    assert response.headers["content-disposition"] == "attachment; filename=speech.mp3"
    # Just verify status and content type
    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/mpeg"
    assert response.headers["content-disposition"] == "attachment; filename=speech.mp3"


@pytest.mark.asyncio
async def test_openai_speech_streaming_generator(mock_tts_service, async_client):
    """Test streaming with async generator"""
    test_request = {
        "model": "kokoro",
        "input": "Hello world",
        "voice": "af",
        "response_format": "pcm",
        "stream": True
    }
    
    # Create streaming mock for this test
    async def mock_stream(*args, **kwargs):
        for chunk in [b"chunk1", b"chunk2"]:
            yield chunk
    mock_tts_service.generate_audio_stream = mock_stream
    
    # Add streaming header
    headers = {"x-raw-response": "stream"}
    response = await async_client.post("/v1/audio/speech", json=test_request, headers=headers)
    
    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/pcm"
    # Just verify status and content type
    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/pcm"
