"""Shim runtime cho RecBole trên NumPy 2.x và PyTorch 2.6+."""

import warnings


def apply_torch_load_compat() -> None:
    """RecBole checkpoint dùng pickle protocol 4 — PyTorch 2.6+ mặc định weights_only=True."""
    import torch

    if getattr(torch.load, "_ncs_patched", False):
        return

    _orig_load = torch.load

    def _load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return _orig_load(*args, **kwargs)

    _load._ncs_patched = True
    torch.load = _load
    # Một số bản RecBole gọi torch.serialization.load trực tiếp
    if hasattr(torch, "serialization") and hasattr(torch.serialization, "load"):
        torch.serialization.load = _load


def apply_numpy_recbole_compat() -> None:
    import numpy as np

    _ALIASES = {
        "bool": np.bool_,
        "int": np.int_,
        "long": np.int_,
        "float": np.float64,
        "float_": np.float64,
        "complex": np.complex128,
        "complex_": np.complex128,
        "unicode": np.str_,
        "unicode_": np.str_,
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        _ALIASES["object"] = np.object_
        _ALIASES["str"] = np.str_

    for name, value in _ALIASES.items():
        if name not in np.__dict__:
            setattr(np, name, value)
