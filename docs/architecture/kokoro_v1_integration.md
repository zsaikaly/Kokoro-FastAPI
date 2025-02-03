# Kokoro V1 Integration Architecture

## Overview

This document outlines the architectural approach for integrating the new Kokoro V1 library into our existing inference system. The goal is to bypass most of the legacy model machinery while maintaining compatibility with our existing interfaces, particularly the OpenAI-compatible streaming endpoint.

## Current System

The current system uses a `ModelBackend` interface with multiple implementations (ONNX CPU/GPU, PyTorch CPU/GPU). This interface requires:

- Async model loading
- Audio generation from tokens and voice tensors
- Resource cleanup
- Device management

## Integration Approach

### 1. KokoroV1 Backend Implementation

We'll create a `KokoroV1` class implementing the `ModelBackend` interface that wraps the new Kokoro library:

```python
class KokoroV1(BaseModelBackend):
    def __init__(self):
        super().__init__()
        self._model = None
        self._pipeline = None
        self._device = "cuda" if settings.use_gpu and torch.cuda.is_available() else "cpu"
```

### 2. Model Loading

The load_model method will initialize both KModel and KPipeline:

```python
async def load_model(self, path: str) -> None:
    model_path = await paths.get_model_path(path)
    self._model = KModel(model_path).to(self._device).eval()
    self._pipeline = KPipeline(model=self._model, device=self._device)
```

### 3. Audio Generation

The generate method will adapt our token/voice tensor format to work with KPipeline:

```python
def generate(self, tokens: list[int], voice: torch.Tensor, speed: float = 1.0) -> np.ndarray:
    # Convert tokens to text using pipeline's tokenizer
    # Use voice tensor as voice embedding
    # Return generated audio
```

### 4. Streaming Support

The Kokoro V1 backend must maintain compatibility with our OpenAI-compatible streaming endpoint. Key requirements:

1. **Chunked Generation**: The pipeline's output should be compatible with our streaming infrastructure:
   ```python
   async def generate_stream(self, text: str, voice_path: str) -> AsyncGenerator[bytes, None]:
       results = self._pipeline(text, voice=voice_path)
       for result in results:
           yield result.audio.numpy()
   ```

2. **Format Conversion**: Support for various output formats:
   - MP3
   - Opus
   - AAC
   - FLAC
   - WAV
   - PCM

3. **Voice Management**:
   - Support for voice combination (mean of multiple voice embeddings)
   - Dynamic voice loading and caching
   - Voice listing and validation

4. **Error Handling**:
   - Proper error propagation for client disconnects
   - Format conversion errors
   - Resource cleanup on failures

### 5. Configuration Integration

We'll use the existing configuration system:

```python
config = model_config.pytorch_kokoro_v1_file  # Model file path
```

## Benefits

1. **Simplified Pipeline**: Direct use of Kokoro library's built-in pipeline
2. **Better Language Support**: Access to Kokoro's wider language capabilities
3. **Automatic Chunking**: Built-in text chunking and processing
4. **Phoneme Generation**: Access to phoneme output for better analysis
5. **Streaming Compatibility**: Maintains existing streaming functionality

## Migration Strategy

1. Implement KokoroV1 backend with streaming support
2. Add to model manager's available backends
3. Make it the default for new requests
4. Keep legacy backends available for backward compatibility
5. Update voice management to handle both legacy and new voice formats

## Next Steps

1. Switch to Code mode to implement the KokoroV1 backend
2. Ensure streaming compatibility with OpenAI endpoint
3. Add tests to verify both streaming and non-streaming functionality
4. Update documentation for new capabilities
5. Add monitoring for streaming performance