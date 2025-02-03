# NLP Dependencies Management

## Overview

This document outlines our approach to managing NLP dependencies, particularly focusing on spaCy models that are required by our dependencies (such as misaki). The goal is to ensure reliable model availability while preventing runtime download attempts that could cause failures.

## Challenge

One of our dependencies, misaki, attempts to download the spaCy model `en_core_web_sm` during runtime. This can lead to failures if:
- The download fails due to network issues
- The environment lacks proper permissions
- The system is running in a restricted environment

## Solution

### Model Management with UV

We use UV (Universal Versioner) as our package manager. For spaCy model management, we have two approaches:

1. **Development Environment Setup**
   ```bash
   uv run --with spacy -- spacy download en_core_web_sm
   ```
   This command:
   - Temporarily installs spaCy if not present
   - Downloads the required model
   - Places it in the appropriate location

2. **Project Environment**
   - Add spaCy as a project dependency in pyproject.toml
   - Run `uv run -- spacy download en_core_web_sm` in the project directory
   - This installs the model in the project's virtual environment

### Docker Environment

For containerized deployments:
1. Add the model download step in the Dockerfile
2. Ensure the model is available before application startup
3. Configure misaki to use the pre-downloaded model

## Benefits

1. **Reliability**: Prevents runtime download attempts
2. **Reproducibility**: Model version is consistent across environments
3. **Performance**: No startup delay from download attempts
4. **Security**: Better control over external downloads

## Implementation Notes

1. Development environments should use the `uv run --with spacy` approach for flexibility
2. CI/CD pipelines should include model download in their setup phase
3. Docker builds should pre-download models during image creation
4. Application code should verify model availability at startup

## Future Considerations

1. Consider caching models in a shared location for multiple services
2. Implement version pinning for NLP models
3. Add health checks to verify model availability
4. Monitor model usage and performance

## Related Documentation

- [Kokoro V1 Integration](kokoro_v1_integration.md)
- UV Package Manager Documentation
- spaCy Model Management Guide