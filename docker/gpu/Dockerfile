FROM --platform=$BUILDPLATFORM nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04
# Set non-interactive frontend
ENV DEBIAN_FRONTEND=noninteractive

# Install Python and other dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-venv \
    espeak-ng \
    git \
    libsndfile1 \
    curl \
    ffmpeg \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Create non-root user
RUN useradd -m -u 1000 appuser

# Create directories and set ownership
RUN mkdir -p /app/api/src/voices && \
    chown -R appuser:appuser /app

USER appuser
WORKDIR /app

# Copy dependency files
COPY --chown=appuser:appuser pyproject.toml ./pyproject.toml

# Install dependencies with GPU extras
RUN --mount=type=cache,target=/root/.cache/uv \
    uv venv && \
    uv sync --extra gpu

# Copy project files including models
COPY --chown=appuser:appuser api ./api
COPY --chown=appuser:appuser web ./web
COPY --chown=appuser:appuser docker/scripts/download_model.* ./

# Install project with GPU extras
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --extra gpu

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV PATH="/app/.venv/bin:$PATH"
ENV UV_LINK_MODE=copy

ENV USE_GPU=true
ENV USE_ONNX=false
ENV DOWNLOAD_PTH=true
ENV DOWNLOAD_ONNX=false

# Download models based on environment variables
RUN if [ "$DOWNLOAD_PTH" = "true" ]; then \
    python download_model.py --type pth; \
    fi && \
    if [ "$DOWNLOAD_ONNX" = "true" ]; then \
    python download_model.py --type onnx; \
    fi

# Run FastAPI server
CMD ["uv", "run", "python", "-m", "uvicorn", "api.src.main:app", "--host", "0.0.0.0", "--port", "8880", "--log-level", "debug"]
