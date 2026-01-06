"""Utility functions for device detection, memory estimation, formatting."""

import gc
import torch
from rich.console import Console

console = Console()


def get_device() -> str:
    """Detect best available device (ROCm-aware)."""
    if torch.cuda.is_available():
        # Works for both CUDA and ROCm (which presents as cuda)
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_memory_info() -> dict:
    """Get available memory info."""
    device = get_device()
    info = {"device": device}
    
    if device == "cuda":
        props = torch.cuda.get_device_properties(0)
        info["total_gb"] = props.total_memory / (1024**3)
        info["name"] = props.name
        # For ROCm/shared memory systems, this might show less than actual
        allocated = torch.cuda.memory_allocated(0) / (1024**3)
        info["allocated_gb"] = allocated
    
    return info


def estimate_model_memory(num_params_b: float, dtype: str = "float16") -> float:
    """Estimate memory needed for model weights in GB.
    
    Args:
        num_params_b: Number of parameters in billions
        dtype: Data type (float16, float32, int8, int4)
    """
    bytes_per_param = {
        "float32": 4,
        "float16": 2,
        "bfloat16": 2,
        "int8": 1,
        "int4": 0.5,
    }
    return num_params_b * bytes_per_param.get(dtype, 2)


def clear_memory():
    """Aggressively clear GPU/CPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def print_model_info(model, tokenizer):
    """Print model configuration info like in the video."""
    config = model.config
    
    console.print(f"[bold]Model:[/bold] {config._name_or_path}")
    console.print(f"[bold]Family:[/bold] {config.model_type}")
    console.print(f"[bold]Layers:[/bold] {config.num_hidden_layers}")
    console.print(f"[bold]Hidden size:[/bold] {config.hidden_size}")
    console.print(f"[bold]Vocab size:[/bold] {config.vocab_size}")
    
    # Check if chat template exists
    has_chat = tokenizer.chat_template is not None
    console.print(f"[bold]Mode:[/bold] {'CHAT (using chat template)' if has_chat else 'COMPLETION'}")
    console.print()
