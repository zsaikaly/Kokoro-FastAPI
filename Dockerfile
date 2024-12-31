# Stage 1: Clone model repository
FROM alpine/git:latest AS model_layer
ARG KOKORO_REPO=https://huggingface.co/hexgrad/Kokoro-82M
ARG KOKORO_COMMIT=a67f11354c3e38c58c3327498bc4bd1e57e71c50

RUN git lfs install --skip-repo
WORKDIR /app/Kokoro-82M
RUN GIT_LFS_SKIP_SMUDGE=1 git clone ${KOKORO_REPO} . && \
    git checkout ${KOKORO_COMMIT} && \
    git lfs pull && \
    ls -la

# Stage 2: Build
FROM nvidia/cuda:12.1.0-base-ubuntu22.04

# Install base system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    python3-dev \
    espeak-ng \
    git \
    libsndfile1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch with CUDA support first
RUN pip3 install --no-cache-dir torch==2.5.1 --extra-index-url https://download.pytorch.org/whl/cu121

# Install all other dependencies from requirements.txt
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Set working directory
WORKDIR /app


# Run with Python unbuffered output for live logging
ENV PYTHONUNBUFFERED=1

# Copy model files from git clone stage
COPY --from=model_layer /app/Kokoro-82M /app/Kokoro-82M

# Create non-root user
RUN useradd -m -u 1000 appuser

# Create and set permissions for output directory
RUN mkdir -p /app/api/src/output && \
    chown -R appuser:appuser /app/api/src/output

# Set Python path (app first for our imports, then model dir for model imports)
ENV PYTHONPATH=/app:/app/Kokoro-82M

# Switch to non-root user
USER appuser

# Run FastAPI server with debug logging and reload
CMD ["uvicorn", "api.src.main:app", "--host", "0.0.0.0", "--port", "8880", "--log-level", "debug"]
