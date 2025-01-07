import os
import sys
import shutil
from unittest.mock import Mock, patch, MagicMock

import pytest
import aiofiles.threadpool


def cleanup_mock_dirs():
    """Clean up any MagicMock directories created during tests"""
    mock_dir = "MagicMock"
    if os.path.exists(mock_dir):
        shutil.rmtree(mock_dir)


@pytest.fixture(autouse=True)
def setup_aiofiles():
    """Setup aiofiles mock wrapper"""
    aiofiles.threadpool.wrap.register(MagicMock)(
        lambda *args, **kwargs: aiofiles.threadpool.AsyncBufferedIOBase(*args, **kwargs)
    )
    yield


@pytest.fixture(autouse=True)
def cleanup():
    """Automatically clean up before and after each test"""
    cleanup_mock_dirs()
    yield
    cleanup_mock_dirs()


# Create mock torch module
mock_torch = Mock()
mock_torch.cuda = Mock()
mock_torch.cuda.is_available = Mock(return_value=False)

# Create a mock tensor class that supports basic operations
class MockTensor:
    def __init__(self, data):
        self.data = data
        if isinstance(data, (list, tuple)):
            self.shape = [len(data)]
        elif isinstance(data, MockTensor):
            self.shape = data.shape
        else:
            self.shape = getattr(data, 'shape', [1])
        
    def __getitem__(self, idx):
        if isinstance(self.data, (list, tuple)):
            if isinstance(idx, slice):
                return MockTensor(self.data[idx])
            return self.data[idx]
        return self
        
    def max(self):
        if isinstance(self.data, (list, tuple)):
            max_val = max(self.data)
            return MockTensor(max_val)
        return 5  # Default for testing
        
    def item(self):
        if isinstance(self.data, (list, tuple)):
            return max(self.data)
        if isinstance(self.data, (int, float)):
            return self.data
        return 5  # Default for testing
        
    def cuda(self):
        """Support cuda conversion"""
        return self
        
    def any(self):
        if isinstance(self.data, (list, tuple)):
            return any(self.data)
        return False
        
    def all(self):
        if isinstance(self.data, (list, tuple)):
            return all(self.data)
        return True
        
    def unsqueeze(self, dim):
        return self
        
    def expand(self, *args):
        return self
        
    def type_as(self, other):
        return self

# Add tensor operations to mock torch
mock_torch.tensor = lambda x: MockTensor(x)
mock_torch.zeros = lambda *args: MockTensor([0] * (args[0] if isinstance(args[0], int) else args[0][0]))
mock_torch.arange = lambda x: MockTensor(list(range(x)))
mock_torch.gt = lambda x, y: MockTensor([False] * x.shape[0])

# Mock modules before they're imported
sys.modules["torch"] = mock_torch
sys.modules["transformers"] = Mock()
sys.modules["phonemizer"] = Mock()
sys.modules["models"] = Mock()
sys.modules["models.build_model"] = Mock()
sys.modules["kokoro"] = Mock()
sys.modules["kokoro.generate"] = Mock()
sys.modules["kokoro.phonemize"] = Mock()
sys.modules["kokoro.tokenize"] = Mock()
sys.modules["onnxruntime"] = Mock()


@pytest.fixture(autouse=True)
def mock_tts_model():
    """Mock TTSModel and TTS model initialization"""
    with patch("api.src.services.tts_model.TTSModel") as mock_tts_model, \
         patch("api.src.services.tts_base.TTSBaseModel") as mock_base_model:
        
        # Mock TTSModel
        model_instance = Mock()
        model_instance.get_instance.return_value = model_instance
        model_instance.get_voicepack.return_value = None
        mock_tts_model.get_instance.return_value = model_instance
        
        # Mock TTS model initialization
        mock_base_model.setup.return_value = 1  # Return dummy voice count
        
        yield model_instance
