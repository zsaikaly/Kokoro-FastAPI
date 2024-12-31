<p align="center">
  <img src="githubbanner.png" alt="Kokoro TTS Banner">
</p>

# Kokoro TTS API
[![Model Commit](https://img.shields.io/badge/model--commit-a67f113-blue)](https://huggingface.co/hexgrad/Kokoro-82M/tree/8228a351f87c8a6076502c1e3b7e72e821ebec9a)
[![Tests](https://img.shields.io/badge/tests-33%20passed-darkgreen)]()
[![Coverage](https://img.shields.io/badge/coverage-97%25-darkgreen)]()

FastAPI wrapper for [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) text-to-speech model, providing an OpenAI-compatible endpoint with:
- NVIDIA GPU accelerated inference (or CPU) option
- automatic chunking/stitching for long texts
- very fast generation time (~35-49x RTF)

## Quick Start

1. Install prerequisites:
   - Install [Docker Desktop](https://www.docker.com/products/docker-desktop/)
   - Install [Git](https://git-scm.com/downloads) (or download and extract zip)

2. Clone and start the service:
```bash
# Clone repository
git clone https://github.com/remsky/Kokoro-FastAPI.git
cd Kokoro-FastAPI

# For GPU acceleration (requires NVIDIA GPU):
docker compose up --build

# For CPU-only deployment (~10x slower, but doesn't require an NVIDIA GPU):
docker compose -f docker-compose.cpu.yml up --build
```



Test all voices (from another terminal):
```bash
python examples/test_all_voices.py
```

Test OpenAI compatibility:
```bash
python examples/test_openai_tts.py
```

## OpenAI-Compatible API

List available voices:
```python
import requests

response = requests.get("http://localhost:8880/audio/voices")
voices = response.json()["voices"]
```

Generate speech:
```python
import requests

response = requests.post(
    "http://localhost:8880/audio/speech",
    json={
        "model": "kokoro",  # Not used but required for compatibility
        "input": "Hello world!",
        "voice": "af_bella",
        "response_format": "mp3",  # Supported: mp3, wav, opus, flac
        "speed": 1.0
    }
)

# Save audio
with open("output.mp3", "wb") as f:
    f.write(response.content)
```

Using OpenAI's Python library:
```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8880", api_key="not-needed")

response = client.audio.speech.create(
    model="kokoro",  # Not used but required for compatibility, also accepts library defaults
    voice="af_bella",
    input="Hello world!",
    response_format="mp3"
)

response.stream_to_file("output.mp3")
```

## Performance Benchmarks

Benchmarking was performed on generation via the local API using text lengths up to feature-length books (~1.5 hours output), measuring processing time and realtime factor. Tests were run on: 
- Windows 11 Home w/ WSL2 
- NVIDIA 4060Ti 16gb GPU @ CUDA 12.1
- 11th Gen i7-11700 @ 2.5GHz
- 64gb RAM
- WAV native output
- H.G. Wells - The Time Machine (full text)

<p align="center">
  <img src="examples/benchmarks/processing_time.png" width="45%" alt="Processing Time" style="border: 2px solid #333; padding: 10px; margin-right: 1%;">
  <img src="examples/benchmarks/realtime_factor.png" width="45%" alt="Realtime Factor" style="border: 2px solid #333; padding: 10px;">
</p>

Key Performance Metrics:
- Realtime Factor: Ranges between 35-49x (generation time to output audio length)
- Average Processing Rate: 137.67 tokens/second

## Features

- OpenAI-compatible API endpoints
- GPU-accelerated inference (if desired)
- Multiple audio formats: mp3, wav, opus, flac, (aac & pcm not implemented)
- Natural Boundary Detection:
    - Automatically splits and stitches at sentence boundaries to reduce artifacts and maintain performacne

*Note: CPU Inference is currently a very basic implementation, and not heavily tested*

## Model

This API uses the [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) model from HuggingFace. 

Visit the model page for more details about training, architecture, and capabilities. I have no affiliation with any of their work, and produced this wrapper for ease of use and personal projects.

## License

This project is licensed under the Apache License 2.0 - see below for details:

- The Kokoro model weights are licensed under Apache 2.0 (see [model page](https://huggingface.co/hexgrad/Kokoro-82M))
- The FastAPI wrapper code in this repository is licensed under Apache 2.0 to match
- The inference code adapted from StyleTTS2 is MIT licensed

The full Apache 2.0 license text can be found at: https://www.apache.org/licenses/LICENSE-2.0

## Sample

<div align="center";">
  
  https://user-images.githubusercontent.com/338912d2-90f3-41fb-bca0-5db7b4e02287.mp4
  
</div>
