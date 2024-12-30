FROM nvidia/cuda:12.1.0-base-ubuntu22.04

ARG KOKORO_REPO

# Install base system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    python3-dev \
    espeak-ng \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install heavy Python dependencies first (better layer caching)
RUN pip3 install --no-cache-dir \
    phonemizer \
    torch \
    transformers \
    scipy \
    munch

# Install API dependencies
RUN pip3 install --no-cache-dir fastapi uvicorn pydantic-settings

# Set working directory
WORKDIR /app

# --(can skip if pre-cloning the repo)--
# Install git-lfs 
RUN apt-get update && apt-get install -y git-lfs \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && git lfs install

# Clone Kokoro repo
RUN git clone ${KOKORO_REPO} .
# --------------------------------------
    
# Create output directory
RUN mkdir -p output

# Run with Python unbuffered output for live logging
ENV PYTHONUNBUFFERED=1

# Copy API files over
COPY api/src /app/api/src

# Set Python path
ENV PYTHONPATH=/app

# Run FastAPI server
CMD ["uvicorn", "api.src.main:app", "--host", "0.0.0.0", "--port", "8880"]
