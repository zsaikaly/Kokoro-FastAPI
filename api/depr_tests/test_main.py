"""Tests for FastAPI application"""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import torch
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
async def test_lifespan_successful_warmup():
    """Test successful model warmup in lifespan"""
    with patch("api.src.inference.model_manager.get_manager") as mock_get_model, \
         patch("api.src.inference.voice_manager.get_manager") as mock_get_voice, \
         patch("api.src.main.logger") as mock_logger, \
         patch("os.path.exists") as mock_exists, \
         patch("torch.cuda.is_available") as mock_cuda:

        # Setup mocks
        mock_model = AsyncMock()
        mock_voice = AsyncMock()
        mock_get_model.return_value = mock_model
        mock_get_voice.return_value = mock_voice
        mock_exists.return_value = True
        mock_cuda.return_value = False

        # Setup model manager
        mock_backend = MagicMock()
        mock_backend.device = "cpu"
        mock_model.get_backend.return_value = mock_backend
        mock_model.load_model = AsyncMock()

        # Setup voice manager
        mock_voice_tensor = torch.zeros(192)
        mock_voice.load_voice = AsyncMock(return_value=mock_voice_tensor)
        mock_voice.list_voices = AsyncMock(return_value=["af", "af_bella", "af_sarah"])

        # Create an async generator from the lifespan context manager
        async_gen = lifespan(MagicMock())
        
        # Start the context manager
        await async_gen.__aenter__()

        # Verify managers were initialized
        mock_get_model.assert_called_once()
        mock_get_voice.assert_called_once()
        mock_model.load_model.assert_called_once()

        # Clean up
        await async_gen.__aexit__(None, None, None)


@pytest.mark.asyncio
async def test_lifespan_failed_warmup():
    """Test failed model warmup in lifespan"""
    with patch("api.src.inference.model_manager.get_manager") as mock_get_model:
        # Mock the model manager to fail
        mock_get_model.side_effect = RuntimeError("Failed to initialize model")

        # Create an async generator from the lifespan context manager
        async_gen = lifespan(MagicMock())

        # Verify the exception is raised
        with pytest.raises(RuntimeError, match="Failed to initialize model"):
            await async_gen.__aenter__()

        # Clean up
        await async_gen.__aexit__(None, None, None)


@pytest.mark.asyncio
async def test_lifespan_voice_manager_failure():
    """Test failure when voice manager fails to initialize"""
    with patch("api.src.inference.model_manager.get_manager") as mock_get_model, \
         patch("api.src.inference.voice_manager.get_manager") as mock_get_voice:

        # Setup model manager success but voice manager failure
        mock_model = AsyncMock()
        mock_get_model.return_value = mock_model
        mock_get_voice.side_effect = RuntimeError("Failed to initialize voice manager")

        # Create an async generator from the lifespan context manager
        async_gen = lifespan(MagicMock())

        # Verify the exception is raised
        with pytest.raises(RuntimeError, match="Failed to initialize voice manager"):
            await async_gen.__aenter__()

        # Clean up
        await async_gen.__aexit__(None, None, None)
