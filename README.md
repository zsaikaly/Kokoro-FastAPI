# Kokoro TTS API

FastAPI wrapper for Kokoro TTS with voice cloning. Runs inference on GPU.

## Quick Start

```bash
# Start the API (will automatically download model on first run)
docker compose up --build
```

```bash
# From host terminal, test it out with some API calls
python examples/test_tts.py "Hello world" --voice af_bella
```
## API Endpoints

```bash
GET /tts/voices           # List voices
POST /tts                 # Generate speech
GET /tts/{request_id}     # Check status
GET /tts/file/{request_id} # Get audio file
```

## Example Usage

List voices:
```bash
python examples/test_tts.py
```

Generate speech:
```bash
# Default voice
python examples/test_tts.py "Your text here"

# Specific voice
python examples/test_tts.py --voice af_bella "Your text here"

# Just get file path (no download)
python examples/test_tts.py --no-download "Your text here"
```

Generated files in `examples/output/` (or in src/output/ of API if --no-download)

## Requirements

- Docker
- NVIDIA GPU + CUDA
- nvidia-container-toolkit
