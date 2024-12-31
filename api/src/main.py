"""
FastAPI OpenAI Compatible API
"""

import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from .core.config import settings
from .routers.openai_compatible import router as openai_router
from .services.tts import TTSModel, TTSService


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for model initialization"""
    logger.info("Loading TTS model and voice packs...")

    # Initialize the main model
    model, device = TTSModel.get_instance()
    logger.info(f"Model loaded on {device}")

    # Initialize all voice packs
    tts_service = TTSService()
    voices = tts_service.list_voices()
    for voice in voices:
        logger.info(f"Loading voice pack: {voice}")
        TTSModel.get_voicepack(voice)

    logger.info("All models and voice packs loaded successfully")
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

# Include OpenAI compatible router
app.include_router(openai_router, prefix="/v1")


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
