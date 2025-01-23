"""Tests for TTSService"""
import os
import numpy as np
import pytest
import torch
from unittest.mock import AsyncMock, MagicMock, Mock, patch

from api.src.services.tts_service import TTSService

# Get project root path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MOCK_VOICES_DIR = os.path.join(PROJECT_ROOT, "api", "src", "voices")
MOCK_MODEL_DIR = os.path.join(PROJECT_ROOT, "api", "src", "models")


@pytest.mark.asyncio
async def test_service_initialization(mock_model_manager, mock_voice_manager):
    """Test TTSService initialization"""
    # Create service using factory method
    with patch("api.src.services.tts_service.get_model_manager", return_value=mock_model_manager), \
         patch("api.src.services.tts_service.get_voice_manager", return_value=mock_voice_manager):
        service = await TTSService.create()
        assert service is not None
        assert service.model_manager == mock_model_manager
        assert service._voice_manager == mock_voice_manager


@pytest.mark.asyncio
async def test_generate_audio_basic(mock_tts_service):
    """Test basic audio generation"""
    text = "Hello world"
    voice = "af"
    audio, duration = await mock_tts_service.generate_audio(text, voice)
    assert isinstance(audio, np.ndarray)
    assert duration > 0


@pytest.mark.asyncio
async def test_generate_audio_empty_text(mock_tts_service):
    """Test handling empty text input"""
    with pytest.raises(ValueError, match="Text is empty after preprocessing"):
        await mock_tts_service.generate_audio("", "af")


@pytest.mark.asyncio
async def test_generate_audio_stream(mock_tts_service):
    """Test streaming audio generation"""
    text = "Hello world"
    voice = "af"
    
    # Setup mock stream
    async def mock_stream():
        yield b"chunk1"
        yield b"chunk2"
    mock_tts_service.generate_audio_stream.return_value = mock_stream()
    
    # Test streaming
    stream = mock_tts_service.generate_audio_stream(text, voice)
    chunks = []
    async for chunk in await stream:
        chunks.append(chunk)
    
    assert len(chunks) > 0
    assert all(isinstance(chunk, bytes) for chunk in chunks)


@pytest.mark.asyncio
async def test_list_voices(mock_tts_service):
    """Test listing available voices"""
    with patch("api.src.inference.voice_manager.settings") as mock_settings:
        mock_settings.voices_dir = MOCK_VOICES_DIR
        voices = await mock_tts_service.list_voices()
        assert isinstance(voices, list)
        assert len(voices) == 4  # ["af", "af_bella", "af_sarah", "bm_lewis"]
        assert all(isinstance(voice, str) for voice in voices)


@pytest.mark.asyncio
async def test_combine_voices(mock_tts_service):
    """Test combining voices"""
    with patch("api.src.inference.voice_manager.settings") as mock_settings:
        mock_settings.voices_dir = MOCK_VOICES_DIR
        voices = ["af_bella", "af_sarah"]
        result = await mock_tts_service.combine_voices(voices)
        assert isinstance(result, str)
        assert result == "af_bella_af_sarah"


@pytest.mark.asyncio
async def test_audio_to_bytes(mock_tts_service):
    """Test converting audio to bytes"""
    audio = np.zeros(48000, dtype=np.float32)
    audio_bytes = mock_tts_service._audio_to_bytes(audio)
    assert isinstance(audio_bytes, bytes)
    assert len(audio_bytes) > 0


@pytest.mark.asyncio
async def test_voice_loading(mock_tts_service):
    """Test voice loading"""
    with patch("api.src.inference.voice_manager.settings") as mock_settings, \
         patch("os.path.exists", return_value=True), \
         patch("torch.load", return_value=torch.zeros(192)):
        mock_settings.voices_dir = MOCK_VOICES_DIR
        voice = await mock_tts_service._voice_manager.load_voice("af", device="cpu")
        assert isinstance(voice, torch.Tensor)
        assert voice.shape == (192,)


@pytest.mark.asyncio
async def test_model_generation(mock_tts_service):
    """Test model generation"""
    tokens = [1, 2, 3]
    voice_tensor = torch.zeros(192)
    audio = await mock_tts_service.model_manager.generate(tokens, voice_tensor)
    assert isinstance(audio, torch.Tensor)
    assert audio.shape == (48000,)
    assert audio.dtype == torch.float32
