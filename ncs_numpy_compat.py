"""Shim runtime cho RecBole trên NumPy 2.x và PyTorch 2.6+."""


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


def apply_numpy_recbole_compat() -> None:
    import numpy as np

    def _set(name: str, value) -> None:
        if not hasattr(np, name):
            setattr(np, name, value)

    _set("bool", np.bool_)
    _set("int", np.int_)
    _set("long", np.int_)
    _set("float", np.float64)
    _set("float_", np.float64)
    _set("complex", np.complex128)
    _set("complex_", np.complex128)
    _set("object", np.object_)
    _set("str", np.str_)
    _set("unicode", np.str_)
    _set("unicode_", np.str_)
