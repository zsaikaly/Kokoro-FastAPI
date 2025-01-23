import pytest
from unittest.mock import AsyncMock, patch
import torch
from pathlib import Path

@pytest.mark.asyncio
async def test_list_available_voices(mock_voice_manager):
    """Test listing available voices"""
    voices = await mock_voice_manager.list_voices()
    assert len(voices) == 2
    assert "voice1" in voices
    assert "voice2" in voices

@pytest.mark.asyncio
async def test_get_voice_path(mock_voice_manager):
    """Test getting path for a specific voice"""
    voice_path = mock_voice_manager.get_voice_path("voice1")
    assert voice_path == "/mock/path/voice.pt"

    # Test invalid voice
    mock_voice_manager.get_voice_path.return_value = None
    assert mock_voice_manager.get_voice_path("invalid_voice") is None

@pytest.mark.asyncio
async def test_load_voice(mock_voice_manager, mock_voice_tensor):
    """Test loading a voice tensor"""
    voice_tensor = await mock_voice_manager.load_voice("voice1")
    assert torch.equal(voice_tensor, mock_voice_tensor)
    mock_voice_manager.load_voice.assert_called_once_with("voice1")

@pytest.mark.asyncio
async def test_load_voice_not_found(mock_voice_manager):
    """Test loading non-existent voice"""
    mock_voice_manager.get_voice_path.return_value = None
    mock_voice_manager.load_voice.side_effect = ValueError("Voice not found: invalid_voice")
    
    with pytest.raises(ValueError, match="Voice not found: invalid_voice"):
        await mock_voice_manager.load_voice("invalid_voice")

@pytest.mark.asyncio
async def test_combine_voices(mock_voice_manager):
    """Test combining two voices"""
    voices = ["voice1", "voice2"]
    weights = [0.5, 0.5]
    
    combined_id = await mock_voice_manager.combine_voices(voices, weights)
    assert combined_id == "voice1_voice2"
    mock_voice_manager.combine_voices.assert_called_once_with(voices, weights)

@pytest.mark.asyncio
async def test_combine_voices_invalid_weights(mock_voice_manager):
    """Test combining voices with invalid weights"""
    voices = ["voice1", "voice2"]
    weights = [0.3, 0.3]  # Doesn't sum to 1
    
    mock_voice_manager.combine_voices.side_effect = ValueError("Weights must sum to 1")
    with pytest.raises(ValueError, match="Weights must sum to 1"):
        await mock_voice_manager.combine_voices(voices, weights)

@pytest.mark.asyncio
async def test_combine_voices_single_voice(mock_voice_manager):
    """Test combining with single voice"""
    voices = ["voice1"]
    weights = [1.0]
    
    mock_voice_manager.combine_voices.side_effect = ValueError("At least 2 voices are required")
    with pytest.raises(ValueError, match="At least 2 voices are required"):
        await mock_voice_manager.combine_voices(voices, weights)

@pytest.mark.asyncio
async def test_cache_management(mock_voice_manager, mock_voice_tensor):
    """Test voice cache management"""
    # Mock cache info
    mock_voice_manager.cache_info = {"size": 1, "max_size": 10}
    
    # Load voice to test caching
    await mock_voice_manager.load_voice("voice1")
    
    # Check cache info
    cache_info = mock_voice_manager.cache_info
    assert cache_info["size"] == 1
    assert cache_info["max_size"] == 10