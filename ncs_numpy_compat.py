"""Shim NumPy 2.x cho RecBole (vẫn gọi np.float_, np.unicode_, ...)."""


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
