"""Tests for API endpoints"""
import pytest
import torch
from fastapi.testclient import TestClient

from ..src.main import app

# Create test client for non-async tests
client = TestClient(app)


def test_health_check():
    """Test the health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


@pytest.mark.asyncio
async def test_openai_speech_endpoint(async_client, mock_tts_service):
    """Test the OpenAI-compatible speech endpoint"""
    # Setup mocks
    mock_tts_service._voice_manager.list_voices.return_value = ["bm_lewis"]
    mock_tts_service.generate_audio.return_value = (torch.zeros(48000).numpy(), 1.0)
    mock_tts_service._voice_manager.load_voice.return_value = torch.zeros(192)

    # Mock voice validation
    mock_tts_service._voice_manager.get_voice_path.return_value = "/mock/voices/bm_lewis.pt"

    test_request = {
        "model": "kokoro",
        "input": "Hello world",
        "voice": "bm_lewis",
        "response_format": "wav",
        "speed": 1.0,
        "stream": False,
    }
    response = await async_client.post("/v1/audio/speech", json=test_request)
    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/wav"
    assert response.headers["content-disposition"] == "attachment; filename=speech.wav"
    mock_tts_service.generate_audio.assert_called_once()


@pytest.mark.asyncio
async def test_openai_speech_invalid_voice(async_client, mock_tts_service):
    """Test the OpenAI-compatible speech endpoint with invalid voice"""
    # Setup mocks
    mock_tts_service._voice_manager.list_voices.return_value = ["af", "bm_lewis"]
    mock_tts_service._voice_manager.get_voice_path.return_value = None

    test_request = {
        "model": "kokoro",
        "input": "Hello world",
        "voice": "invalid_voice",
        "response_format": "wav",
        "speed": 1.0,
        "stream": False,
    }
    response = await async_client.post("/v1/audio/speech", json=test_request)
    assert response.status_code == 400
    assert "not found" in response.json()["detail"]["message"]


@pytest.mark.asyncio
async def test_openai_speech_generation_error(async_client, mock_tts_service):
    """Test error handling in speech generation"""
    # Setup mocks
    mock_tts_service._voice_manager.list_voices.return_value = ["af"]
    mock_tts_service.generate_audio.side_effect = RuntimeError("Generation failed")
    mock_tts_service._voice_manager.load_voice.return_value = torch.zeros(192)
    mock_tts_service._voice_manager.get_voice_path.return_value = "/mock/voices/af.pt"

    test_request = {
        "model": "kokoro",
        "input": "Hello world",
        "voice": "af",
        "response_format": "wav",
        "speed": 1.0,
        "stream": False,
    }
    response = await async_client.post("/v1/audio/speech", json=test_request)
    assert response.status_code == 500
    assert "Generation failed" in response.json()["detail"]["message"]


@pytest.mark.asyncio
async def test_combine_voices_list_success(async_client, mock_tts_service):
    """Test successful voice combination using list format"""
    # Setup mocks
    mock_tts_service._voice_manager.list_voices.return_value = ["af_bella", "af_sarah"]
    mock_tts_service._voice_manager.combine_voices.return_value = "af_bella_af_sarah"
    mock_tts_service._voice_manager.load_voice.return_value = torch.zeros(192)
    mock_tts_service._voice_manager.get_voice_path.return_value = "/mock/voices/af_bella.pt"

    test_voices = ["af_bella", "af_sarah"]
    response = await async_client.post("/v1/audio/voices/combine", json=test_voices)

    assert response.status_code == 200
    assert response.json()["voice"] == "af_bella_af_sarah"
    mock_tts_service._voice_manager.combine_voices.assert_called_once()


@pytest.mark.asyncio
async def test_combine_voices_empty_list(async_client, mock_tts_service):
    """Test combining empty voice list returns error"""
    test_voices = []
    response = await async_client.post("/v1/audio/voices/combine", json=test_voices)
    assert response.status_code == 400
    assert "No voices provided" in response.json()["detail"]["message"]


@pytest.mark.asyncio
async def test_speech_streaming_with_combined_voice(async_client, mock_tts_service):
    """Test streaming speech with combined voice using + syntax"""
    # Setup mocks
    mock_tts_service._voice_manager.list_voices.return_value = ["af_bella", "af_sarah"]
    mock_tts_service._voice_manager.combine_voices.return_value = "af_bella_af_sarah"
    mock_tts_service._voice_manager.load_voice.return_value = torch.zeros(192)
    mock_tts_service._voice_manager.get_voice_path.return_value = "/mock/voices/af_bella.pt"

    async def mock_stream():
        yield b"chunk1"
        yield b"chunk2"

    mock_tts_service.generate_audio_stream.return_value = mock_stream()

    test_request = {
        "model": "kokoro",
        "input": "Hello world",
        "voice": "af_bella+af_sarah",
        "response_format": "mp3",
        "stream": True,
    }

    headers = {"x-raw-response": "stream"}
    response = await async_client.post(
        "/v1/audio/speech", json=test_request, headers=headers
    )

    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/mpeg"
    assert response.headers["content-disposition"] == "attachment; filename=speech.mp3"


@pytest.mark.asyncio
async def test_openai_speech_pcm_streaming(async_client, mock_tts_service):
    """Test streaming PCM audio for real-time playback"""
    # Setup mocks
    mock_tts_service._voice_manager.list_voices.return_value = ["af"]
    mock_tts_service._voice_manager.load_voice.return_value = torch.zeros(192)
    mock_tts_service._voice_manager.get_voice_path.return_value = "/mock/voices/af.pt"

    async def mock_stream():
        yield b"chunk1"
        yield b"chunk2"

    mock_tts_service.generate_audio_stream.return_value = mock_stream()

    test_request = {
        "model": "kokoro",
        "input": "Hello world",
        "voice": "af",
        "response_format": "pcm",
        "stream": True,
    }

    headers = {"x-raw-response": "stream"}
    response = await async_client.post(
        "/v1/audio/speech", json=test_request, headers=headers
    )

    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/pcm"


@pytest.mark.asyncio
async def test_openai_speech_streaming_mp3(async_client, mock_tts_service):
    """Test streaming MP3 audio to file"""
    # Setup mocks
    mock_tts_service._voice_manager.list_voices.return_value = ["af"]
    mock_tts_service._voice_manager.load_voice.return_value = torch.zeros(192)
    mock_tts_service._voice_manager.get_voice_path.return_value = "/mock/voices/af.pt"

    async def mock_stream():
        yield b"chunk1"
        yield b"chunk2"

    mock_tts_service.generate_audio_stream.return_value = mock_stream()

    test_request = {
        "model": "kokoro",
        "input": "Hello world",
        "voice": "af",
        "response_format": "mp3",
        "stream": True,
    }

    headers = {"x-raw-response": "stream"}
    response = await async_client.post(
        "/v1/audio/speech", json=test_request, headers=headers
    )

    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/mpeg"
    assert response.headers["content-disposition"] == "attachment; filename=speech.mp3"
