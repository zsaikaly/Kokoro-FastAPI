from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # API Settings
    api_title: str = "Kokoro TTS API"
    api_description: str = "API for text-to-speech generation using Kokoro"
    api_version: str = "1.0.0"
    host: str = "0.0.0.0"
    port: int = 8880

    # TTS Settings
    output_dir: str = "output"
    output_dir_size_limit_mb: float = 500.0  # Maximum size of output directory in MB
    default_voice: str = "af"
    use_gpu: bool = False  # Whether to use GPU acceleration if available
    use_onnx: bool = True  # Whether to use ONNX runtime
    # Container absolute paths
    model_dir: str = "/app/api/src/models"  # Absolute path in container
    voices_dir: str = "/app/api/src/voices"  # Absolute path in container
    
    # Model filenames
    pytorch_model_file: str = "kokoro-v0_19.pth"
    onnx_model_file: str = "kokoro-v0_19.onnx"
    sample_rate: int = 24000
    max_chunk_size: int = 300  # Maximum size of text chunks for processing
    gap_trim_ms: int = 250  # Amount to trim from streaming chunk ends in milliseconds

    # ONNX Optimization Settings
    onnx_num_threads: int = 4  # Number of threads for intra-op parallelism
    onnx_inter_op_threads: int = 4  # Number of threads for inter-op parallelism
    onnx_execution_mode: str = "parallel"  # parallel or sequential
    onnx_optimization_level: str = "all"  # all, basic, or disabled
    onnx_memory_pattern: bool = True  # Enable memory pattern optimization
    onnx_arena_extend_strategy: str = "kNextPowerOfTwo"  # Memory allocation strategy
    
    # ONNX GPU Settings
    onnx_device_id: int = 0  # GPU device ID to use
    onnx_gpu_mem_limit: float = 0.7  # Limit GPU memory usage to 70%
    onnx_cudnn_conv_algo_search: str = "EXHAUSTIVE"  # CUDNN convolution algorithm search
    onnx_do_copy_in_default_stream: bool = True  # Copy in default CUDA stream

    class Config:
        env_file = ".env"


settings = Settings()
