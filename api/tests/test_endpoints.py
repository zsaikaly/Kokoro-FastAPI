from fastapi.testclient import TestClient
import pytest
from unittest.mock import Mock, patch
from ..src.main import app
from ..src.services.tts import TTSService

# Create test client
client = TestClient(app)

# Mock TTSService methods
@pytest.fixture
def mock_tts_service():
    with patch("api.src.routers.tts.TTSService") as mock_service:
        # Setup mock returns
        service_instance = Mock()
        service_instance.list_voices.return_value = ["af", "en"]
        service_instance.create_tts_request.return_value = 1
        service_instance.get_request_status.return_value = Mock(
            id=1,
            status="completed",
            output_file="test.wav",
            processing_time=1.0
        )
        mock_service.return_value = service_instance
        yield service_instance

def test_health_check():
    """Test the health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

def test_list_voices(mock_tts_service):
    """Test listing available voices"""
    response = client.get("/tts/voices")
    assert response.status_code == 200
    assert response.json() == {
        "voices": ["af", "en"],
        "default": "af"
    }

def test_create_tts_request(mock_tts_service):
    """Test creating a TTS request"""
    test_request = {
        "text": "Hello world",
        "voice": "af",
        "speed": 1.0,
        "stitch_long_output": True
    }
    response = client.post("/tts", json=test_request)
    assert response.status_code == 200
    assert response.json() == {
        "request_id": 1,
        "status": "pending",
        "output_file": None,
        "processing_time": None
    }

def test_create_tts_invalid_voice(mock_tts_service):
    """Test creating a TTS request with invalid voice"""
    test_request = {
        "text": "Hello world",
        "voice": "invalid_voice",
        "speed": 1.0,
        "stitch_long_output": True
    }
    response = client.post("/tts", json=test_request)
    assert response.status_code == 400
    assert "Voice 'invalid_voice' not found" in response.json()["detail"]

def test_get_tts_status(mock_tts_service):
    """Test getting TTS request status"""
    response = client.get("/tts/1")
    assert response.status_code == 200
    assert response.json() == {
        "request_id": 1,
        "status": "completed",
        "output_file": "test.wav",
        "processing_time": 1.0
    }

def test_get_tts_status_not_found(mock_tts_service):
    """Test getting status of non-existent request"""
    mock_tts_service.get_request_status.return_value = None
    response = client.get("/tts/999")
    assert response.status_code == 404
    assert response.json()["detail"] == "Request not found"

@patch("builtins.open", create=True)
@patch("os.path.exists", return_value=True)
def test_get_audio_file(mock_exists, mock_open, mock_tts_service):
    """Test downloading audio file"""
    # Set up mock request status with output file
    mock_request = Mock(
        id=1,
        status="completed",  # Must match the status check in router
        output_file="test.wav",
        processing_time=1.0
    )
    mock_tts_service.get_request_status.return_value = mock_request
    
    # Mock file read
    mock_open.return_value.__enter__.return_value.read.return_value = b"audio data"
    
    response = client.get("/tts/file/1")
    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/wav"
    assert response.headers["content-disposition"] == "attachment; filename=speech_1.wav"
    assert response.content == b"audio data"

def test_get_audio_file_not_found(mock_tts_service):
    """Test downloading non-existent audio file"""
    mock_tts_service.get_request_status.return_value = None
    response = client.get("/tts/file/999")
    assert response.status_code == 404
    assert response.json()["detail"] == "Request not found"

def test_create_tts_invalid_speed(mock_tts_service):
    """Test creating a TTS request with invalid speed"""
    test_request = {
        "text": "Hello world",
        "voice": "af",
        "speed": -1.0,  # Invalid speed
        "stitch_long_output": True
    }
    response = client.post("/tts", json=test_request)
    assert response.status_code == 422  # Validation error

def test_get_audio_file_not_completed(mock_tts_service):
    """Test getting audio file for request that's still processing"""
    mock_request = Mock(
        id=1,
        status="processing",  # Not completed yet
        output_file=None,
        processing_time=None
    )
    mock_tts_service.get_request_status.return_value = mock_request
    
    response = client.get("/tts/file/1")
    assert response.status_code == 400
    assert response.json()["detail"] == "Audio generation not complete"

def test_get_tts_status_processing(mock_tts_service):
    """Test getting status of a processing request"""
    mock_request = Mock(
        id=1,
        status="processing",
        output_file=None,
        processing_time=None
    )
    mock_tts_service.get_request_status.return_value = mock_request
    
    response = client.get("/tts/1")
    assert response.status_code == 200
    assert response.json() == {
        "request_id": 1,
        "status": "processing",
        "output_file": None,
        "processing_time": None
    }
