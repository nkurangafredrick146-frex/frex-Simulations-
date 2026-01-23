"""GPU acceleration helpers (placeholder).

This module should be replaced or extended with CUDA/Numba/OpenCL helpers.
"""
def supports_cuda():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False

__all__ = ["supports_cuda"]
