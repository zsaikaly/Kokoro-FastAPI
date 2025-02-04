"""Model configuration for Kokoro V1."""

from pydantic import BaseModel, Field

class KokoroV1Config(BaseModel):
    """Kokoro V1 configuration."""
    languages: list[str] = ["en"]

    class Config:
        frozen = True

class PyTorchCPUConfig(BaseModel):
    """PyTorch CPU backend configuration."""
    
    memory_threshold: float = Field(0.8, description="Memory threshold for cleanup")
    retry_on_oom: bool = Field(True, description="Whether to retry on OOM errors")
    num_threads: int = Field(8, description="Number of threads for parallel operations")
    pin_memory: bool = Field(True, description="Whether to pin memory for faster CPU-GPU transfer")

    class Config:
        frozen = True


class PyTorchGPUConfig(BaseModel):
    """PyTorch GPU backend configuration."""
    
    device_id: int = Field(0, description="CUDA device ID")
    use_triton: bool = Field(True, description="Whether to use Triton for CUDA kernels")
    memory_threshold: float = Field(0.8, description="Memory threshold for cleanup")
    retry_on_oom: bool = Field(True, description="Whether to retry on OOM errors")
    sync_cuda: bool = Field(True, description="Whether to synchronize CUDA operations")
    cuda_streams: int = Field(2, description="Number of CUDA streams for inference")
    stream_timeout: int = Field(60, description="Stream timeout in seconds")

    class Config:
        frozen = True


class ModelConfig(BaseModel):
    """Kokoro V1 model configuration."""
    
    # General settings
    device_type: str = Field("cpu", description="Device type ('cpu' or 'gpu')")
    cache_voices: bool = Field(True, description="Whether to cache voice tensors")
    voice_cache_size: int = Field(2, description="Maximum number of cached voices")
    
    # Model filename
    pytorch_kokoro_v1_file: str = Field("v1_0/kokoro-v1_0.pth", description="PyTorch Kokoro V1 model filename")
    
    # Backend configs
    pytorch_cpu: PyTorchCPUConfig = Field(default_factory=PyTorchCPUConfig)
    pytorch_gpu: PyTorchGPUConfig = Field(default_factory=PyTorchGPUConfig)

    class Config:
        frozen = True


# Global instance
model_config = ModelConfig()