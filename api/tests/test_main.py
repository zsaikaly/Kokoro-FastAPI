"""Tests for main FastAPI application"""
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from api.src.main import app


@pytest.fixture
def client():
    """Create a test client"""
    return TestClient(app)


def test_health_check(client):
    """Test health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_test_endpoint(client):
    """Test the test endpoint"""
    response = client.get("/v1/test")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_cors_headers(client):
    """Test CORS headers are present"""
    response = client.get(
        "/health",
        headers={"Origin": "http://testserver"},
    )
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "*"


def test_openapi_schema(client):
    """Test OpenAPI schema is accessible"""
    response = client.get("/openapi.json")
    assert response.status_code == 200
    schema = response.json()
    assert schema["info"]["title"] == app.title
    assert schema["info"]["version"] == app.version
