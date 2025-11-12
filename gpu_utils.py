"""Small GPU utility: expose xp (cupy if available and requested) and helpers.

Usage:
    from gpu_utils import xp, set_device, to_cpu

Call set_device('gpu') early (before heavy imports) to request CuPy.
If CuPy isn't available or device is not 'gpu', xp will be NumPy.
"""
from typing import Any
import os

_use_gpu = False
_xp = None

def _try_use_cupy():
    try:
        import cupy as cp
        return cp
    except Exception:
        return None

def set_device(device: str):
    """Set desired device: 'gpu' or 'cpu'. This sets the module-level xp alias.

    Call this before importing modules that create large arrays (e.g. in `main.py`).
    """
    global _use_gpu, _xp
    if device and device.lower() in ('gpu', 'cuda'):
        cp = _try_use_cupy()
        if cp is not None:
            _use_gpu = True
            _xp = cp
            return
    # fallback to numpy
    import numpy as np
    _use_gpu = False
    _xp = np

def use_gpu() -> bool:
    return _use_gpu

@property
def xp():
    """Return the current array module (CuPy or NumPy)."""
    global _xp
    if _xp is None:
        # default: respect environment variable GPU=1 to attempt GPU
        if os.environ.get('GPU', '0') == '1':
            set_device('gpu')
        else:
            set_device('cpu')
    return _xp

def to_cpu(arr: Any):
    """Convert an array to a NumPy array on host.

    If xp is NumPy this is a no-op.
    """
    if use_gpu():
        try:
            return arr.get() if hasattr(arr, 'get') else arr
        except Exception:
            # fall back
            import numpy as np
            return np.array(arr)
    else:
        return arr
