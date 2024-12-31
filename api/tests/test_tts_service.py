"""Tests for TTSService"""
import os
import numpy as np
import pytest
from unittest.mock import patch, MagicMock, call
from api.src.services.tts import TTSService, TTSModel


@pytest.fixture
def tts_service():
    """Create a TTSService instance for testing"""
    return TTSService(start_worker=False)


@pytest.fixture
def sample_audio():
    """Generate a simple sine wave for testing"""
    sample_rate = 24000
    duration = 0.1  # 100ms
    t = np.linspace(0, duration, int(sample_rate * duration))
    frequency = 440  # A4 note
    return np.sin(2 * np.pi * frequency * t).astype(np.float32)


def test_split_text(tts_service):
    """Test text splitting into sentences"""
    text = "First sentence. Second sentence! Third sentence?"
    sentences = tts_service._split_text(text)
    assert len(sentences) == 3
    assert sentences[0] == "First sentence."
    assert sentences[1] == "Second sentence!"
    assert sentences[2] == "Third sentence?"


def test_split_text_empty(tts_service):
    """Test splitting empty text"""
    assert tts_service._split_text("") == []


def test_split_text_single_sentence(tts_service):
    """Test splitting single sentence"""
    text = "Just one sentence."
    assert tts_service._split_text(text) == ["Just one sentence."]


def test_audio_to_bytes(tts_service, sample_audio):
    """Test converting audio tensor to bytes"""
    audio_bytes = tts_service._audio_to_bytes(sample_audio)
    assert isinstance(audio_bytes, bytes)
    assert len(audio_bytes) > 0


@patch('os.listdir')
@patch('os.path.join')
def test_list_voices(mock_join, mock_listdir, tts_service):
    """Test listing available voices"""
    mock_listdir.return_value = ['voice1.pt', 'voice2.pt', 'not_a_voice.txt']
    mock_join.return_value = '/fake/path'
    
    voices = tts_service.list_voices()
    assert len(voices) == 2
    assert 'voice1' in voices
    assert 'voice2' in voices
    assert 'not_a_voice' not in voices


@patch('api.src.services.tts.TTSModel.get_instance')
@patch('api.src.services.tts.TTSModel.get_voicepack')
@patch('api.src.services.tts.normalize_text')
@patch('api.src.services.tts.phonemize')
@patch('api.src.services.tts.tokenize')
@patch('api.src.services.tts.generate')
def test_generate_audio_empty_text(mock_generate, mock_tokenize, mock_phonemize, mock_normalize, mock_voicepack, mock_instance, tts_service):
    """Test generating audio with empty text"""
    mock_normalize.return_value = ""
    
    with pytest.raises(ValueError, match="Text is empty after preprocessing"):
        tts_service._generate_audio("", "af", 1.0)


@patch('api.src.services.tts.TTSModel.get_instance')
@patch('api.src.services.tts.TTSModel.get_voicepack')
@patch('api.src.services.tts.normalize_text')
@patch('api.src.services.tts.phonemize')
@patch('api.src.services.tts.tokenize')
@patch('api.src.services.tts.generate')
def test_generate_audio_no_chunks(mock_generate, mock_tokenize, mock_phonemize, mock_normalize, mock_voicepack, mock_instance, tts_service):
    """Test generating audio with no successful chunks"""
    mock_normalize.return_value = "Test text"
    mock_phonemize.return_value = "Test text"
    mock_tokenize.return_value = ["test", "text"]
    mock_generate.return_value = (None, None)
    mock_instance.return_value = (MagicMock(), "cpu")
    
    with pytest.raises(ValueError, match="No audio chunks were generated successfully"):
        tts_service._generate_audio("Test text", "af", 1.0)


@patch('api.src.services.tts.TTSModel.get_instance')
@patch('api.src.services.tts.TTSModel.get_voicepack')
@patch('api.src.services.tts.normalize_text')
@patch('api.src.services.tts.phonemize')
@patch('api.src.services.tts.tokenize')
@patch('api.src.services.tts.generate')
def test_generate_audio_success(mock_generate, mock_tokenize, mock_phonemize, mock_normalize, mock_voicepack, mock_instance, tts_service, sample_audio):
    """Test successful audio generation"""
    mock_normalize.return_value = "Test text"
    mock_phonemize.return_value = "Test text"
    mock_tokenize.return_value = ["test", "text"]
    mock_generate.return_value = (sample_audio, None)
    mock_instance.return_value = (MagicMock(), "cpu")
    
    audio, processing_time = tts_service._generate_audio("Test text", "af", 1.0)
    assert isinstance(audio, np.ndarray)
    assert isinstance(processing_time, float)
    assert len(audio) > 0


@patch('api.src.services.tts.torch.cuda.is_available')
@patch('api.src.services.tts.build_model')
def test_model_initialization_cuda(mock_build_model, mock_cuda_available):
    """Test model initialization with CUDA"""
    mock_cuda_available.return_value = True
    mock_model = MagicMock()
    mock_build_model.return_value = mock_model

    TTSModel._instance = None  # Reset singleton
    model, device = TTSModel.get_instance()
    
    assert device == "cuda"
    assert model == mock_model
    mock_build_model.assert_called_once()


@patch('api.src.services.tts.torch.cuda.is_available')
@patch('api.src.services.tts.build_model')
def test_model_initialization_cpu(mock_build_model, mock_cuda_available):
    """Test model initialization with CPU"""
    mock_cuda_available.return_value = False
    mock_model = MagicMock()
    mock_build_model.return_value = mock_model

    TTSModel._instance = None  # Reset singleton
    model, device = TTSModel.get_instance()
    
    assert device == "cpu"
    assert model == mock_model
    mock_build_model.assert_called_once()


@patch('api.src.services.tts.torch.load')
@patch('os.path.join')
def test_voicepack_loading_error(mock_join, mock_torch_load):
    """Test voicepack loading error handling"""
    mock_join.side_effect = lambda *args: '/'.join(args)
    mock_torch_load.side_effect = [Exception("Failed to load voice"), MagicMock()]
    
    TTSModel._instance = (MagicMock(), "cpu")  # Mock instance
    TTSModel._voicepacks = {}  # Reset voicepacks
    
    # Should fall back to 'af' voice
    voicepack = TTSModel.get_voicepack("nonexistent_voice")
    assert mock_torch_load.call_count == 2  # Tried original voice then fallback
    assert isinstance(voicepack, MagicMock)  # Successfully got fallback voice


@patch('api.src.services.tts.torch.load')
@patch('os.path.join')
def test_voicepack_loading_error_af(mock_join, mock_torch_load):
    """Test voicepack loading error for 'af' voice"""
    mock_join.side_effect = lambda *args: '/'.join(args)
    mock_torch_load.side_effect = Exception("Failed to load voice")
    
    TTSModel._instance = (MagicMock(), "cpu")  # Mock instance
    TTSModel._voicepacks = {}  # Reset voicepacks
    
    with pytest.raises(Exception):
        TTSModel.get_voicepack("af")


def test_save_audio(tts_service, sample_audio, tmp_path):
    """Test saving audio to file"""
    output_path = os.path.join(tmp_path, "test_output", "audio.wav")
    tts_service._save_audio(sample_audio, output_path)
    
    assert os.path.exists(output_path)
    assert os.path.getsize(output_path) > 0


@patch('api.src.services.tts.TTSModel.get_instance')
@patch('api.src.services.tts.TTSModel.get_voicepack')
@patch('api.src.services.tts.normalize_text')
@patch('api.src.services.tts.generate')
def test_generate_audio_without_stitching(mock_generate, mock_normalize, mock_voicepack, mock_instance, tts_service, sample_audio):
    """Test generating audio without text stitching"""
    mock_normalize.return_value = "Test text"
    mock_generate.return_value = (sample_audio, None)
    mock_instance.return_value = (MagicMock(), "cpu")
    
    audio, processing_time = tts_service._generate_audio("Test text", "af", 1.0, stitch_long_output=False)
    assert isinstance(audio, np.ndarray)
    assert isinstance(processing_time, float)
    assert len(audio) > 0
    mock_generate.assert_called_once()


@patch('os.listdir')
def test_list_voices_error(mock_listdir, tts_service):
    """Test error handling in list_voices"""
    mock_listdir.side_effect = Exception("Failed to list directory")
    
    voices = tts_service.list_voices()
    assert voices == []


@patch('api.src.services.tts.TTSModel.get_instance')
@patch('api.src.services.tts.TTSModel.get_voicepack')
@patch('api.src.services.tts.normalize_text')
@patch('api.src.services.tts.phonemize')
@patch('api.src.services.tts.tokenize')
@patch('api.src.services.tts.generate')
def test_generate_audio_phonemize_error(mock_generate, mock_tokenize, mock_phonemize, mock_normalize, mock_voicepack, mock_instance, tts_service):
    """Test handling phonemization error"""
    mock_normalize.return_value = "Test text"
    mock_phonemize.side_effect = Exception("Phonemization failed")
    mock_instance.return_value = (MagicMock(), "cpu")
    mock_generate.return_value = (None, None)
    
    with pytest.raises(ValueError, match="No audio chunks were generated successfully"):
        tts_service._generate_audio("Test text", "af", 1.0)


@patch('api.src.services.tts.TTSModel.get_instance')
@patch('api.src.services.tts.TTSModel.get_voicepack')
@patch('api.src.services.tts.normalize_text')
@patch('api.src.services.tts.generate')
def test_generate_audio_error(mock_generate, mock_normalize, mock_voicepack, mock_instance, tts_service):
    """Test handling generation error"""
    mock_normalize.return_value = "Test text"
    mock_generate.side_effect = Exception("Generation failed")
    mock_instance.return_value = (MagicMock(), "cpu")
    
    with pytest.raises(ValueError, match="No audio chunks were generated successfully"):
        tts_service._generate_audio("Test text", "af", 1.0)
