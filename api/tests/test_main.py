"""Tests for FastAPI application"""
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from api.src.main import app, lifespan


@pytest.fixture
def test_client():
    """Create a test client"""
    return TestClient(app)


def test_health_check(test_client):
    """Test health check endpoint"""
    response = test_client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


@pytest.mark.asyncio
@patch('api.src.main.TTSModel')
@patch('api.src.main.logger')
async def test_lifespan_successful_warmup(mock_logger, mock_tts_model):
    """Test successful model warmup in lifespan"""
    # Mock the model initialization with model info and voicepack count
    mock_model = MagicMock()
    # Mock file system for voice counting
    mock_tts_model.VOICES_DIR = "/mock/voices"
    with patch('os.listdir', return_value=['voice1.pt', 'voice2.pt', 'voice3.pt']):
        mock_tts_model.initialize.return_value = (mock_model, 3)  # 3 voice files
        mock_tts_model._device = "cuda"  # Set device class variable
    
    # Create an async generator from the lifespan context manager
    async_gen = lifespan(MagicMock())
    # Start the context manager
    await async_gen.__aenter__()
    
    # Verify the expected logging sequence
    mock_logger.info.assert_any_call("Loading TTS model and voice packs...")
    mock_logger.info.assert_any_call("Model loaded and warmed up on cuda")
    mock_logger.info.assert_any_call("3 voice packs loaded successfully")
    
    # Verify model initialization was called
    mock_tts_model.initialize.assert_called_once()
    
    # Clean up
    await async_gen.__aexit__(None, None, None)


@pytest.mark.asyncio
@patch('api.src.main.TTSModel')
@patch('api.src.main.logger')
async def test_lifespan_failed_warmup(mock_logger, mock_tts_model):
    """Test failed model warmup in lifespan"""
    # Mock the model initialization to fail
    mock_tts_model.initialize.side_effect = Exception("Failed to initialize model")
    
    # Create an async generator from the lifespan context manager
    async_gen = lifespan(MagicMock())
    
    # Verify the exception is raised
    with pytest.raises(Exception, match="Failed to initialize model"):
        await async_gen.__aenter__()
    
    # Verify the expected logging sequence
    mock_logger.info.assert_called_with("Loading TTS model and voice packs...")
    
    # Clean up
    await async_gen.__aexit__(None, None, None)


@pytest.mark.asyncio
@patch('api.src.main.TTSModel')
async def test_lifespan_cuda_warmup(mock_tts_model):
    """Test model warmup specifically on CUDA"""
    # Mock the model initialization with CUDA and voicepacks
    mock_model = MagicMock()
    # Mock file system for voice counting
    mock_tts_model.VOICES_DIR = "/mock/voices"
    with patch('os.listdir', return_value=['voice1.pt', 'voice2.pt']):
        mock_tts_model.initialize.return_value = (mock_model, 2)  # 2 voice files
        mock_tts_model._device = "cuda"  # Set device class variable
    
    # Create an async generator from the lifespan context manager
    async_gen = lifespan(MagicMock())
    await async_gen.__aenter__()
    
    # Verify model was initialized
    mock_tts_model.initialize.assert_called_once()
    
    # Clean up
    await async_gen.__aexit__(None, None, None)


@pytest.mark.asyncio
@patch('api.src.main.TTSModel')
async def test_lifespan_cpu_fallback(mock_tts_model):
    """Test model warmup falling back to CPU"""
    # Mock the model initialization with CPU and voicepacks
    mock_model = MagicMock()
    # Mock file system for voice counting
    mock_tts_model.VOICES_DIR = "/mock/voices"
    with patch('os.listdir', return_value=['voice1.pt', 'voice2.pt', 'voice3.pt', 'voice4.pt']):
        mock_tts_model.initialize.return_value = (mock_model, 4)  # 4 voice files
        mock_tts_model._device = "cpu"  # Set device class variable
    
    # Create an async generator from the lifespan context manager
    async_gen = lifespan(MagicMock())
    await async_gen.__aenter__()
    
    # Verify model was initialized
    mock_tts_model.initialize.assert_called_once()
    
    # Clean up
    await async_gen.__aexit__(None, None, None)
