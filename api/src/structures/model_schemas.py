"""Model and voice configuration schemas."""

from pydantic import BaseModel


class ModelConfig(BaseModel):
    """Model configuration."""
    optimization_level: str = "all"  # all, basic, none
    num_threads: int = 4
    inter_op_threads: int = 4
    execution_mode: str = "parallel"  # parallel, sequential
    memory_pattern: bool = True
    arena_extend_strategy: str = "kNextPowerOfTwo"

    class Config:
        frozen = True  # Make config immutable


class VoiceConfig(BaseModel):
    """Voice configuration."""
    use_cache: bool = True
    cache_size: int = 3  # Number of voices to cache
    validate_on_load: bool = True  # Whether to validate voices when loading

    class Config:
        frozen = True  # Make config immutable