"""Tests for AudioService"""
import numpy as np
import pytest
from api.src.services.audio import AudioService


@pytest.fixture
def sample_audio():
    """Generate a simple sine wave for testing"""
    sample_rate = 24000
    duration = 0.1  # 100ms
    t = np.linspace(0, duration, int(sample_rate * duration))
    frequency = 440  # A4 note
    return np.sin(2 * np.pi * frequency * t).astype(np.float32), sample_rate


def test_convert_to_wav(sample_audio):
    """Test converting to WAV format"""
    audio_data, sample_rate = sample_audio
    result = AudioService.convert_audio(audio_data, sample_rate, "wav")
    assert isinstance(result, bytes)
    assert len(result) > 0


def test_convert_to_mp3(sample_audio):
    """Test converting to MP3 format"""
    audio_data, sample_rate = sample_audio
    result = AudioService.convert_audio(audio_data, sample_rate, "mp3")
    assert isinstance(result, bytes)
    assert len(result) > 0


def test_convert_to_opus(sample_audio):
    """Test converting to Opus format"""
    audio_data, sample_rate = sample_audio
    result = AudioService.convert_audio(audio_data, sample_rate, "opus")
    assert isinstance(result, bytes)
    assert len(result) > 0


def test_convert_to_flac(sample_audio):
    """Test converting to FLAC format"""
    audio_data, sample_rate = sample_audio
    result = AudioService.convert_audio(audio_data, sample_rate, "flac")
    assert isinstance(result, bytes)
    assert len(result) > 0


def test_convert_to_aac_raises_error(sample_audio):
    """Test that converting to AAC raises an error"""
    audio_data, sample_rate = sample_audio
    with pytest.raises(ValueError, match="AAC format is not currently supported"):
        AudioService.convert_audio(audio_data, sample_rate, "aac")


def test_convert_to_pcm_raises_error(sample_audio):
    """Test that converting to PCM raises an error"""
    audio_data, sample_rate = sample_audio
    with pytest.raises(ValueError, match="PCM format is not currently supported"):
        AudioService.convert_audio(audio_data, sample_rate, "pcm")


def test_convert_to_invalid_format_raises_error(sample_audio):
    """Test that converting to an invalid format raises an error"""
    audio_data, sample_rate = sample_audio
    with pytest.raises(ValueError, match="Format invalid not supported"):
        AudioService.convert_audio(audio_data, sample_rate, "invalid")
