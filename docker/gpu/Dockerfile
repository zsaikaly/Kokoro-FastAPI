FROM --platform=$BUILDPLATFORM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# Set non-interactive frontend
ENV DEBIAN_FRONTEND=noninteractive

# Install Python and other dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-venv \
    espeak-ng \
    git \
    libsndfile1 \
    curl \
    ffmpeg \
 && ls -la /usr/lib/x86_64-linux-gnu/espeak-ng-data \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

# Create user and set up permissions 
RUN useradd -m -u 1000 appuser && \
    mkdir -p /app/api/src/models/v1_0 && \
    chown -R appuser:appuser /app && \
    chown -R appuser:appuser /usr/lib/x86_64-linux-gnu/espeak-ng-data


# Rest of your Dockerfile...

# Install UV in a separate step
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    mv /root/.local/bin/uv /usr/local/bin/ && \
    mv /root/.local/bin/uvx /usr/local/bin/

USER appuser
WORKDIR /app
# Copy dependency files
COPY --chown=appuser:appuser pyproject.toml ./pyproject.toml

# Install dependencies with GPU extras (using cache mounts)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv venv && \
    uv sync --extra gpu

# Copy project files including models and sync again
COPY --chown=appuser:appuser api ./api
COPY --chown=appuser:appuser web ./web
COPY --chown=appuser:appuser docker/scripts/download_model.* ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --extra gpu

# Set all environment variables in one go
ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app:/app/api \
    PATH="/app/.venv/bin:$PATH" \
    UV_LINK_MODE=copy \
    USE_GPU=true
    
ENV DOWNLOAD_MODEL=true
# Download model if enabled
RUN if [ "$DOWNLOAD_MODEL" = "true" ]; then \
    python download_model.py --output api/src/models/v1_0; \
    fi

# Run FastAPI server
CMD ["uv", "run", "python", "-m", "uvicorn", "api.src.main:app", "--host", "0.0.0.0", "--port", "8880", "--log-level", "debug"]
