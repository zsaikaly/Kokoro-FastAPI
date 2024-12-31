import pytest
from unittest.mock import Mock, patch
import sys

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
