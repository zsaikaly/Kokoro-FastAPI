"""Tests for text processing endpoints"""
import os
import pytest
import torch
from fastapi.testclient import TestClient

from ..src.main import app

# Get project root path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MOCK_VOICES_DIR = os.path.join(PROJECT_ROOT, "api", "src", "voices")

client = TestClient(app)


@pytest.mark.asyncio
async def test_generate_from_phonemes(async_client, mock_tts_service):
    """Test generating audio from phonemes"""
    # Setup mocks
    mock_tts_service._voice_manager.list_voices.return_value = ["af_bella"]
    mock_tts_service.generate_audio.return_value = (torch.zeros(48000).numpy(), 1.0)
    mock_tts_service._voice_manager.load_voice.return_value = torch.zeros(192)
    mock_tts_service._voice_manager.get_voice_path.return_value = "/mock/voices/af_bella.pt"

    test_request = {
        "model": "kokoro",
        "input": "h @ l oU w r= l d",
        "voice": "af_bella",
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
async def test_generate_from_phonemes_invalid_voice(async_client, mock_tts_service):
    """Test generating audio from phonemes with invalid voice"""
    # Setup mocks
    mock_tts_service._voice_manager.list_voices.return_value = ["af_bella"]
    mock_tts_service._voice_manager.get_voice_path.return_value = None

    test_request = {
        "model": "kokoro",
        "input": "h @ l oU w r= l d",
        "voice": "invalid_voice",
        "response_format": "wav",
        "speed": 1.0,
        "stream": False,
    }

    response = await async_client.post("/v1/audio/speech", json=test_request)
    assert response.status_code == 400
    assert "Voice not found" in response.json()["detail"]["message"]


@pytest.mark.asyncio
async def test_generate_from_phonemes_chunked(async_client, mock_tts_service):
    """Test generating chunked audio from phonemes"""
    # Setup mocks
    mock_tts_service._voice_manager.list_voices.return_value = ["af_bella"]
    mock_tts_service._voice_manager.load_voice.return_value = torch.zeros(192)
    mock_tts_service._voice_manager.get_voice_path.return_value = "/mock/voices/af_bella.pt"

    async def mock_stream():
        yield b"chunk1"
        yield b"chunk2"

    mock_tts_service.generate_audio_stream.return_value = mock_stream()

    test_request = {
        "model": "kokoro",
        "input": "h @ l oU w r= l d",
        "voice": "af_bella",
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
async def test_invalid_phonemes(async_client, mock_tts_service):
    """Test handling invalid phonemes"""
    # Setup mocks
    mock_tts_service._voice_manager.list_voices.return_value = ["af_bella"]
    mock_tts_service._voice_manager.load_voice.return_value = torch.zeros(192)
    mock_tts_service._voice_manager.get_voice_path.return_value = "/mock/voices/af_bella.pt"

    test_request = {
        "model": "kokoro",
        "input": "",  # Empty input
        "voice": "af_bella",
        "response_format": "wav",
        "speed": 1.0,
        "stream": False,
    }

    response = await async_client.post("/v1/audio/speech", json=test_request)
    assert response.status_code == 400
    assert "Text is empty" in response.json()["detail"]["message"]


@pytest.mark.asyncio
async def test_phonemes_with_combined_voice(async_client, mock_tts_service):
    """Test generating audio from phonemes with combined voice"""
    # Setup mocks
    mock_tts_service._voice_manager.list_voices.return_value = ["af_bella", "af_sarah"]
    mock_tts_service._voice_manager.combine_voices.return_value = "af_bella_af_sarah"
    mock_tts_service._voice_manager.load_voice.return_value = torch.zeros(192)
    mock_tts_service._voice_manager.get_voice_path.return_value = "/mock/voices/af_bella_af_sarah.pt"
    mock_tts_service.generate_audio.return_value = (torch.zeros(48000).numpy(), 1.0)

    test_request = {
        "model": "kokoro",
        "input": "h @ l oU w r= l d",
        "voice": "af_bella+af_sarah",
        "response_format": "wav",
        "speed": 1.0,
        "stream": False,
    }

    response = await async_client.post("/v1/audio/speech", json=test_request)
    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/wav"
    mock_tts_service._voice_manager.combine_voices.assert_called_once()
    mock_tts_service.generate_audio.assert_called_once()
