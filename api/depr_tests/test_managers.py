"""Tests for model and voice managers"""
import os
import numpy as np
import pytest
import torch
from unittest.mock import AsyncMock, MagicMock, Mock, patch

from api.src.inference.model_manager import get_manager as get_model_manager
from api.src.inference.voice_manager import get_manager as get_voice_manager

# Get project root path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MOCK_VOICES_DIR = os.path.join(PROJECT_ROOT, "api", "src", "voices")
MOCK_MODEL_DIR = os.path.join(PROJECT_ROOT, "api", "src", "models")


@pytest.mark.asyncio
async def test_model_manager_initialization():
    """Test model manager initialization"""
    with patch("api.src.inference.model_manager.settings") as mock_settings, \
         patch("api.src.core.paths.get_model_path") as mock_get_path:

        mock_settings.model_dir = MOCK_MODEL_DIR
        mock_settings.onnx_model_path = "model.onnx"
        mock_get_path.return_value = os.path.join(MOCK_MODEL_DIR, "model.onnx")

        manager = await get_model_manager()
        assert manager is not None
        backend = manager.get_backend()
        assert backend is not None


@pytest.mark.asyncio
async def test_model_manager_generate():
    """Test model generation"""
    with patch("api.src.inference.model_manager.settings") as mock_settings, \
         patch("api.src.core.paths.get_model_path") as mock_get_path, \
         patch("torch.load") as mock_torch_load:

        mock_settings.model_dir = MOCK_MODEL_DIR
        mock_settings.onnx_model_path = "model.onnx"
        mock_settings.use_onnx = True
        mock_settings.use_gpu = False
        mock_get_path.return_value = os.path.join(MOCK_MODEL_DIR, "model.onnx")

        # Mock torch load to return a tensor
        mock_torch_load.return_value = torch.zeros(192)

        manager = await get_model_manager()
        
        # Set up mock backend
        mock_backend = AsyncMock()
        mock_backend.is_loaded = True
        mock_backend.device = "cpu"

        # Create audio tensor and ensure it's properly mocked
        audio_data = torch.zeros(48000, dtype=torch.float32)
        async def mock_generate(*args, **kwargs):
            return audio_data
        mock_backend.generate.side_effect = mock_generate

        # Set up manager with mock backend
        manager._backends['onnx_cpu'] = mock_backend
        manager._current_backend = 'onnx_cpu'

        # Generate audio
        tokens = [1, 2, 3]
        voice_tensor = torch.zeros(192)
        audio = await manager.generate(tokens, voice_tensor, speed=1.0)

        assert isinstance(audio, torch.Tensor), "Generated audio should be torch tensor"
        assert audio.dtype == torch.float32, "Audio should be 32-bit float"
        assert audio.shape == (48000,), "Audio should have 48000 samples"
        assert mock_backend.generate.call_count == 1


@pytest.mark.asyncio
async def test_voice_manager_initialization():
    """Test voice manager initialization"""
    with patch("api.src.inference.voice_manager.settings") as mock_settings, \
         patch("os.path.exists") as mock_exists:
        
        mock_settings.voices_dir = MOCK_VOICES_DIR
        mock_exists.return_value = True

        manager = await get_voice_manager()
        assert manager is not None


@pytest.mark.asyncio
async def test_voice_manager_list_voices():
    """Test listing available voices"""
    with patch("api.src.inference.voice_manager.settings") as mock_settings, \
         patch("os.listdir") as mock_listdir, \
         patch("os.makedirs") as mock_makedirs, \
         patch("os.path.exists") as mock_exists:

        mock_settings.voices_dir = MOCK_VOICES_DIR
        mock_listdir.return_value = ["af_bella.pt", "af_sarah.pt", "bm_lewis.pt"]
        mock_exists.return_value = True

        manager = await get_voice_manager()
        voices = await manager.list_voices()

        assert isinstance(voices, list)
        assert len(voices) == 3, f"Expected 3 voices but got {len(voices)}"
        assert sorted(voices) == ["af_bella", "af_sarah", "bm_lewis"]
        mock_listdir.assert_called_once()


@pytest.mark.asyncio
async def test_voice_manager_load_voice():
    """Test loading a voice"""
    with patch("api.src.inference.voice_manager.settings") as mock_settings, \
         patch("torch.load") as mock_torch_load, \
         patch("os.path.exists") as mock_exists:

        mock_settings.voices_dir = MOCK_VOICES_DIR
        mock_exists.return_value = True
        
        # Create a mock tensor
        mock_tensor = torch.zeros(192)
        mock_torch_load.return_value = mock_tensor

        manager = await get_voice_manager()
        voice_tensor = await manager.load_voice("af_bella", device="cpu")

        assert isinstance(voice_tensor, torch.Tensor)
        assert voice_tensor.shape == (192,)
        mock_torch_load.assert_called_once()


@pytest.mark.asyncio
async def test_voice_manager_combine_voices():
    """Test combining voices"""
    with patch("api.src.inference.voice_manager.settings") as mock_settings, \
         patch("torch.load") as mock_load, \
         patch("torch.save") as mock_save, \
         patch("os.makedirs") as mock_makedirs, \
         patch("os.path.exists") as mock_exists:

        mock_settings.voices_dir = MOCK_VOICES_DIR
        mock_exists.return_value = True

        # Create mock tensors
        mock_tensor1 = torch.ones(192)
        mock_tensor2 = torch.ones(192) * 2
        mock_load.side_effect = [mock_tensor1, mock_tensor2]

        manager = await get_voice_manager()
        combined_name = await manager.combine_voices(["af_bella", "af_sarah"])

        assert combined_name == "af_bella_af_sarah"
        assert mock_load.call_count == 2
        mock_save.assert_called_once()

        # Verify the combined tensor was saved
        saved_tensor = mock_save.call_args[0][0]
        assert isinstance(saved_tensor, torch.Tensor)
        assert saved_tensor.shape == (192,)
        # Should be average of the two tensors
        assert torch.allclose(saved_tensor, torch.ones(192) * 1.5)


@pytest.mark.asyncio
async def test_voice_manager_invalid_voice():
    """Test loading invalid voice"""
    with patch("api.src.inference.voice_manager.settings") as mock_settings, \
         patch("os.path.exists") as mock_exists:

        mock_settings.voices_dir = MOCK_VOICES_DIR
        mock_exists.return_value = False

        manager = await get_voice_manager()
        with pytest.raises(RuntimeError, match="Voice not found"):
            await manager.load_voice("invalid_voice", device="cpu")


@pytest.mark.asyncio
async def test_voice_manager_combine_invalid_voices():
    """Test combining with invalid voices"""
    with patch("api.src.inference.voice_manager.settings") as mock_settings, \
         patch("os.path.exists") as mock_exists:

        mock_settings.voices_dir = MOCK_VOICES_DIR
        mock_exists.return_value = False

        manager = await get_voice_manager()
        with pytest.raises(RuntimeError, match="Voice not found"):
            await manager.combine_voices(["invalid_voice1", "invalid_voice2"])