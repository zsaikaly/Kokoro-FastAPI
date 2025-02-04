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
 && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install UV using the installer script
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    mv /root/.local/bin/uv /usr/local/bin/ && \
    mv /root/.local/bin/uvx /usr/local/bin/

# Create non-root user and prepare /app in one layer
RUN useradd -m -u 1000 appuser && \
    mkdir -p /app/api/src/voices/v1_0 && \
    chown -R appuser:appuser /app

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

# Copy scripts and make them executable in a single RUN step
COPY --chown=appuser:appuser docker/scripts/ /app/docker/scripts/
RUN chmod +x docker/scripts/entrypoint.sh docker/scripts/download_model.sh

# Set all environment variables in one go
ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app:/app/api \
    PATH="/app/.venv/bin:$PATH" \
    UV_LINK_MODE=copy \
    USE_GPU=true \
    USE_ONNX=false \
    DOWNLOAD_PTH=true \
    DOWNLOAD_ONNX=false

# Run FastAPI server
CMD ["/app/docker/scripts/entrypoint.sh"]
