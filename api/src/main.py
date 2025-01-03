"""
FastAPI OpenAI Compatible API
"""

from contextlib import asynccontextmanager

import uvicorn
from loguru import logger
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .core.config import settings
from .services.tts_model import TTSModel
from .services.tts_service import TTSService
from .routers.openai_compatible import router as openai_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for model initialization"""
    logger.info("Loading TTS model and voice packs...")

    # Initialize the main model with warm-up
    voicepack_count = TTSModel.initialize()
    logger.info(f"Model loaded and warmed up on {TTSModel._device}")
    logger.info(f"{voicepack_count} voice packs loaded successfully")
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
