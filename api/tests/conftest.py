import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import numpy as np
import torch
from pathlib import Path

from api.src.services.tts_service import TTSService
from api.src.inference.voice_manager import VoiceManager
from api.src.inference.model_manager import ModelManager
from api.src.structures.model_schemas import VoiceConfig

@pytest.fixture
def mock_voice_tensor():
    """Mock voice tensor for testing."""
    return torch.randn(1, 128)  # Dummy tensor

@pytest.fixture
def mock_audio_output():
    """Mock audio output for testing."""
    return np.random.rand(16000)  # 1 second of random audio

@pytest_asyncio.fixture
async def mock_model_manager(mock_audio_output):
    """Mock model manager for testing."""
    manager = AsyncMock(spec=ModelManager)
    manager.get_backend = MagicMock()
    manager.generate = AsyncMock(return_value=mock_audio_output)
    return manager

@pytest_asyncio.fixture
async def mock_voice_manager(mock_voice_tensor):
    """Mock voice manager for testing."""
    manager = AsyncMock(spec=VoiceManager)
    manager.get_voice_path = MagicMock(return_value="/mock/path/voice.pt")
    manager.load_voice = AsyncMock(return_value=mock_voice_tensor)
    manager.list_voices = AsyncMock(return_value=["voice1", "voice2"])
    manager.combine_voices = AsyncMock(return_value="voice1_voice2")
    return manager

@pytest_asyncio.fixture
async def tts_service(mock_model_manager, mock_voice_manager):
    """Get mocked TTS service instance."""
    service = TTSService()
    service.model_manager = mock_model_manager
    service._voice_manager = mock_voice_manager
    return service

@pytest.fixture
def test_voice():
    """Return a test voice name."""
    return "voice1"

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    import asyncio
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    yield loop
    loop.close()
