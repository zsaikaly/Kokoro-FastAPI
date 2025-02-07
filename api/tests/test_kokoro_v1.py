import pytest
from unittest.mock import patch, MagicMock
import torch
import numpy as np
from api.src.inference.kokoro_v1 import KokoroV1

@pytest.fixture
def kokoro_backend():
    """Create a KokoroV1 instance for testing."""
    return KokoroV1()

def test_initial_state(kokoro_backend):
    """Test initial state of KokoroV1."""
    assert not kokoro_backend.is_loaded
    assert kokoro_backend._model is None
    assert kokoro_backend._pipeline is None
    # Device should be set based on settings
    assert kokoro_backend.device in ["cuda", "cpu"]

@patch('torch.cuda.is_available', return_value=True)
@patch('torch.cuda.memory_allocated')
def test_memory_management(mock_memory, mock_cuda, kokoro_backend):
    """Test GPU memory management functions."""
    # Mock GPU memory usage
    mock_memory.return_value = 5e9  # 5GB
    
    # Test memory check
    with patch('api.src.inference.kokoro_v1.model_config') as mock_config:
        mock_config.pytorch_gpu.memory_threshold = 4
        assert kokoro_backend._check_memory() == True
        
        mock_config.pytorch_gpu.memory_threshold = 6
        assert kokoro_backend._check_memory() == False

@patch('torch.cuda.empty_cache')
@patch('torch.cuda.synchronize')
def test_clear_memory(mock_sync, mock_clear, kokoro_backend):
    """Test memory clearing."""
    with patch.object(kokoro_backend, '_device', 'cuda'):
        kokoro_backend._clear_memory()
        mock_clear.assert_called_once()
        mock_sync.assert_called_once()

@pytest.mark.asyncio
async def test_load_model_validation(kokoro_backend):
    """Test model loading validation."""
    with pytest.raises(RuntimeError, match="Failed to load Kokoro model"):
        await kokoro_backend.load_model("nonexistent_model.pth")

def test_unload(kokoro_backend):
    """Test model unloading."""
    # Mock loaded state
    kokoro_backend._model = MagicMock()
    kokoro_backend._pipeline = MagicMock()
    assert kokoro_backend.is_loaded
    
    # Test unload
    kokoro_backend.unload()
    assert not kokoro_backend.is_loaded
    assert kokoro_backend._model is None
    assert kokoro_backend._pipeline is None

@pytest.mark.asyncio
async def test_generate_validation(kokoro_backend):
    """Test generation validation."""
    with pytest.raises(RuntimeError, match="Model not loaded"):
        async for _ in kokoro_backend.generate("test", "voice"):
            pass

@pytest.mark.asyncio
async def test_generate_from_tokens_validation(kokoro_backend):
    """Test token generation validation."""
    with pytest.raises(RuntimeError, match="Model not loaded"):
        async for _ in kokoro_backend.generate_from_tokens("test tokens", "voice"):
            pass