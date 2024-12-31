from fastapi.testclient import TestClient
import pytest
from unittest.mock import Mock, patch
from ..src.main import app
from ..src.services.tts import TTSService
from ..src.routers.openai_compatible import TTSService as OpenAITTSService

# Create test client
client = TestClient(app)

# Mock services
@pytest.fixture
def mock_tts_service(monkeypatch):
    mock_service = Mock()
    mock_service._generate_audio.return_value = (bytes([0, 1, 2, 3]), 1.0)
    mock_service.list_voices.return_value = ["af", "bm_lewis", "bf_isabella", "bf_emma", "af_sarah", "af_bella", "am_adam", "am_michael", "bm_george"]
    monkeypatch.setattr("api.src.routers.openai_compatible.TTSService", lambda *args, **kwargs: mock_service)
    return mock_service

@pytest.fixture
def mock_audio_service(monkeypatch):
    def mock_convert(*args):
        return b"converted mock audio data"
    monkeypatch.setattr("api.src.routers.openai_compatible.AudioService.convert_audio", mock_convert)

def test_health_check():
    """Test the health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

def test_openai_speech_endpoint(mock_tts_service, mock_audio_service):
    """Test the OpenAI-compatible speech endpoint"""
    test_request = {
        "model": "tts-1",
        "input": "Hello world",
        "voice": "bm_lewis",
        "response_format": "wav",
        "speed": 1.0
    }
    response = client.post("/v1/audio/speech", json=test_request)
    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/wav"
    assert response.headers["content-disposition"] == "attachment; filename=speech.wav"
    mock_tts_service._generate_audio.assert_called_once_with(
        text="Hello world",
        voice="bm_lewis",
        speed=1.0,
        stitch_long_output=True
    )
    assert response.content == b"converted mock audio data"

def test_openai_speech_invalid_voice(mock_tts_service):
    """Test the OpenAI-compatible speech endpoint with invalid voice"""
    test_request = {
        "model": "tts-1",
        "input": "Hello world",
        "voice": "invalid_voice",
        "response_format": "wav",
        "speed": 1.0
    }
    response = client.post("/v1/audio/speech", json=test_request)
    assert response.status_code == 422  # Validation error

def test_openai_speech_invalid_speed(mock_tts_service):
    """Test the OpenAI-compatible speech endpoint with invalid speed"""
    test_request = {
        "model": "tts-1",
        "input": "Hello world",
        "voice": "af",
        "response_format": "wav",
        "speed": -1.0  # Invalid speed
    }
    response = client.post("/v1/audio/speech", json=test_request)
    assert response.status_code == 422  # Validation error

def test_openai_speech_generation_error(mock_tts_service):
    """Test error handling in speech generation"""
    mock_tts_service._generate_audio.side_effect = Exception("Generation failed")
    test_request = {
        "model": "tts-1",
        "input": "Hello world",
        "voice": "af",
        "response_format": "wav",
        "speed": 1.0
    }
    response = client.post("/v1/audio/speech", json=test_request)
    assert response.status_code == 500
    assert "Generation failed" in response.json()["detail"]
