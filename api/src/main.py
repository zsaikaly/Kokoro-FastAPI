import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .core.config import settings
from .routers import tts_router
from .database.database import init_db
from .services.tts import TTSModel

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for database and model initialization"""
    print("Initializing database and preloading models...")
    init_db()  # Initialize database tables
    
    # Preload TTS model and default voice
    TTSModel.get_instance()  # This loads the model
    TTSModel.get_voicepack("af")  # Preload default voice, optional
    print("Initialization complete!")
    
    yield

# Initialize FastAPI app
app = FastAPI(
    title=settings.api_title,
    description=settings.api_description,
    version=settings.api_version,
    lifespan=lifespan
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
app.include_router(tts_router)


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}


if __name__ == "__main__":
    uvicorn.run("api.src.main:app", host=settings.host, port=settings.port, reload=True)
