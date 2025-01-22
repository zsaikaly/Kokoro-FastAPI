
"""
FastAPI OpenAI Compatible API
"""

import os
import sys
from contextlib import asynccontextmanager

import torch
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from .core.config import settings
from .routers.development import router as dev_router
from .routers.openai_compatible import router as openai_router
from .services.tts_service import TTSService


def setup_logger():
    """Configure loguru logger with custom formatting"""
    config = {
        "handlers": [
            {
                "sink": sys.stdout,
                "format": "<fg #2E8B57>{time:hh:mm:ss A}</fg #2E8B57> | "
                "{level: <8} | "
                "{message}",
                "colorize": True,
                "level": "INFO",
            },
        ],
    }
    logger.remove()
    logger.configure(**config)
    logger.level("ERROR", color="<red>")


# Configure logger
setup_logger()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for model initialization"""
    from .inference.model_manager import get_manager
    from .inference.voice_manager import get_manager as get_voice_manager

    logger.info("Loading TTS model and voice packs...")

    try:
        # Initialize managers globally
        model_manager = await get_manager()
        voice_manager = await get_voice_manager()

        # Determine backend type based on settings
        if settings.use_gpu and torch.cuda.is_available():
            backend_type = 'pytorch_gpu' if not settings.use_onnx else 'onnx_gpu'
        else:
            backend_type = 'pytorch_cpu' if not settings.use_onnx else 'onnx_cpu'

        # Get backend and initialize model
        backend = model_manager.get_backend(backend_type)
        
        # Use model path directly from settings
        model_file = settings.pytorch_model_file if not settings.use_onnx else settings.onnx_model_file
        model_path = os.path.join(settings.model_dir, model_file)
        
        
        if not os.path.exists(model_path):
            raise RuntimeError(f"Model file not found: {model_path}")

        # Pre-cache default voice and use for warmup
        warmup_voice = await voice_manager.load_voice(settings.default_voice, device=backend.device)
        logger.info(f"Pre-cached voice {settings.default_voice} for warmup")
        
        # Initialize model with warmup voice
        await model_manager.load_model(model_path, warmup_voice, backend_type)

        # Pre-cache common voices in background
        common_voices = ['af', 'af_bella', 'af_sarah', 'af_nicole']
        for voice_name in common_voices:
            try:
                await voice_manager.load_voice(voice_name, device=backend.device)
                logger.debug(f"Pre-cached voice {voice_name}")
            except Exception as e:
                logger.warning(f"Failed to pre-cache voice {voice_name}: {e}")

        # Get available voices for startup message
        voices = await voice_manager.list_voices()
        voicepack_count = len(voices)

        # Get device info for startup message
        device = "GPU" if settings.use_gpu else "CPU"
        model = "ONNX" if settings.use_onnx else "PyTorch"
    except Exception as e:
        logger.error(f"Failed to initialize model: {e}")
        raise
    boundary = "░" * 2*12
    startup_msg = f"""

{boundary}

    ╔═╗┌─┐┌─┐┌┬┐
    ╠╣ ├─┤└─┐ │
    ╚  ┴ ┴└─┘ ┴
    ╦╔═┌─┐┬┌─┌─┐
    ╠╩╗│ │├┴┐│ │
    ╩ ╩└─┘┴ ┴└─┘

{boundary}
                """
    startup_msg += f"\nModel warmed up on {device}: {model}"
    startup_msg += f"\n{voicepack_count} voice packs loaded\n"
    startup_msg += f"\n{boundary}\n"
    logger.info(startup_msg)

    yield


# Initialize FastAPI app
app = FastAPI(
    title=settings.api_title,
    description=settings.api_description,
    version=settings.api_version,
    lifespan=lifespan,
    openapi_url="/openapi.json",  # Explicitly enable OpenAPI schema
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(openai_router, prefix="/v1")
app.include_router(dev_router)  # New development endpoints
# app.include_router(text_router)  # Deprecated but still live for backwards compatibility


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}


@app.get("/v1/test")
async def test_endpoint():
    """Test endpoint to verify routing"""
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run("api.src.main:app", host=settings.host, port=settings.port, reload=True)
