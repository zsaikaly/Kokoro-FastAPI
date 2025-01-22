from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # API Settings
    api_title: str = "Kokoro TTS API"
    api_description: str = "API for text-to-speech generation using Kokoro"
    api_version: str = "1.0.0"
    host: str = "0.0.0.0"
    port: int = 8880

    # Application Settings
    output_dir: str = "output"
    output_dir_size_limit_mb: float = 500.0  # Maximum size of output directory in MB
    default_voice: str = "af"
    use_gpu: bool = False  # Whether to use GPU acceleration if available
    use_onnx: bool = True  # Whether to use ONNX runtime
    
    # Container absolute paths
    model_dir: str = "/app/api/src/models"  # Absolute path in container
    voices_dir: str = "/app/api/src/voices"  # Absolute path in container
    
    # Audio Settings
    sample_rate: int = 24000
    max_chunk_size: int = 300  # Maximum size of text chunks for processing
    gap_trim_ms: int = 250  # Amount to trim from streaming chunk ends in milliseconds

    class Config:
        env_file = ".env"


settings = Settings()
