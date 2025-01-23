#!/bin/bash
set -e

# Get version from argument or use default
VERSION=${1:-"latest"}

# GitHub Container Registry settings
REGISTRY="ghcr.io"
OWNER="remsky"
REPO="kokoro-fastapi"

# Build CPU image
echo "Building CPU image..."
docker build -t ${REGISTRY}/${OWNER}/${REPO}-cpu:${VERSION} -f docker/cpu/Dockerfile .
docker tag ${REGISTRY}/${OWNER}/${REPO}-cpu:${VERSION} ${REGISTRY}/${OWNER}/${REPO}-cpu:latest

# Build GPU image
echo "Building GPU image..."
docker build -t ${REGISTRY}/${OWNER}/${REPO}-gpu:${VERSION} -f docker/gpu/Dockerfile .
docker tag ${REGISTRY}/${OWNER}/${REPO}-gpu:${VERSION} ${REGISTRY}/${OWNER}/${REPO}-gpu:latest

echo "Build complete!"
echo "Created images:"
echo "- ${REGISTRY}/${OWNER}/${REPO}-cpu:${VERSION}"
echo "- ${REGISTRY}/${OWNER}/${REPO}-cpu:latest"
echo "- ${REGISTRY}/${OWNER}/${REPO}-gpu:${VERSION}"
echo "- ${REGISTRY}/${OWNER}/${REPO}-gpu:latest"

echo -e "\nTo push to GitHub Container Registry:"
echo "docker push ${REGISTRY}/${OWNER}/${REPO}-cpu:${VERSION}"
echo "docker push ${REGISTRY}/${OWNER}/${REPO}-cpu:latest"
echo "docker push ${REGISTRY}/${OWNER}/${REPO}-gpu:${VERSION}"
echo "docker push ${REGISTRY}/${OWNER}/${REPO}-gpu:latest"
