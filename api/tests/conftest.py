import pytest
from unittest.mock import Mock, patch
import sys
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Mock torch and other ML modules before they're imported
sys.modules['torch'] = Mock()
sys.modules['transformers'] = Mock()
sys.modules['phonemizer'] = Mock()
sys.modules['models'] = Mock()
sys.modules['models.build_model'] = Mock()
sys.modules['kokoro'] = Mock()
sys.modules['kokoro.generate'] = Mock()
sys.modules['kokoro.phonemize'] = Mock()
sys.modules['kokoro.tokenize'] = Mock()

from api.src.database.database import Base, get_db
from api.src.main import app

# Use SQLite for testing
SQLALCHEMY_DATABASE_URL = "sqlite:///./test.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

@pytest.fixture(scope="function")
def db():
    """Create a fresh database for each test"""
    Base.metadata.create_all(bind=engine)
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()
        Base.metadata.drop_all(bind=engine)

@pytest.fixture(scope="function")
def client(db):
    """Create a test client with database dependency override"""
    def override_get_db():
        try:
            yield db
        finally:
            db.close()
    
    app.dependency_overrides[get_db] = override_get_db
    yield app.dependency_overrides
    app.dependency_overrides = {}

@pytest.fixture(autouse=True)
def mock_tts_model():
    """Mock TTSModel to avoid loading real models during tests"""
    with patch("api.src.services.tts.TTSModel") as mock:
        model_instance = Mock()
        model_instance.get_instance.return_value = model_instance
        model_instance.get_voicepack.return_value = None
        mock.get_instance.return_value = model_instance
        yield model_instance
