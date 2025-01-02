import os
import shutil
import sys
from unittest.mock import Mock, patch

import pytest

def cleanup_mock_dirs():
    """Clean up any MagicMock directories created during tests"""
    mock_dir = "MagicMock"
    if os.path.exists(mock_dir):
        shutil.rmtree(mock_dir)

@pytest.fixture(autouse=True)
def cleanup():
    """Automatically clean up before and after each test"""
    cleanup_mock_dirs()
    yield
    cleanup_mock_dirs()

# Mock torch and other ML modules before they're imported
sys.modules["torch"] = Mock()
sys.modules["transformers"] = Mock()
sys.modules["phonemizer"] = Mock()
sys.modules["models"] = Mock()
sys.modules["models.build_model"] = Mock()
sys.modules["kokoro"] = Mock()
sys.modules["kokoro.generate"] = Mock()
sys.modules["kokoro.phonemize"] = Mock()
sys.modules["kokoro.tokenize"] = Mock()


@pytest.fixture(autouse=True)
def mock_tts_model():
    """Mock TTSModel to avoid loading real models during tests"""
    with patch("api.src.services.tts.TTSModel") as mock:
        model_instance = Mock()
        model_instance.get_instance.return_value = model_instance
        model_instance.get_voicepack.return_value = None
        mock.get_instance.return_value = model_instance
        yield model_instance
