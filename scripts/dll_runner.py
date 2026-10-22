"""Dynamic Library (DLL) ctypes runner for low-latency native C math routines targeting Windows (.dll)."""

from __future__ import annotations

import ctypes
import os
from typing import List, Optional

from scripts.compile_native import compile_native_library


class NativeBridgeRunner:
    """Wrapper for calling C DLL functions via ctypes."""

    def __init__(self, lib_path: Optional[str] = None):
        if lib_path is None:
            default_path = os.path.join("build", "native_bridge.dll")
            if not os.path.exists(default_path):
                default_path = compile_native_library()
            lib_path = default_path

        self.lib_path = os.path.normpath(lib_path)
        if not os.path.exists(self.lib_path):
            raise FileNotFoundError(f"Native DLL library not found at: {self.lib_path}")

        self.lib = ctypes.CDLL(self.lib_path)
        self._setup_function_signatures()

    def _setup_function_signatures(self) -> None:
        # double fast_dot_product(const double* a, const double* b, int size)
        self.lib.fast_dot_product.argtypes = [
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_int,
        ]
        self.lib.fast_dot_product.restype = ctypes.c_double

        # void fast_vector_scale(const double* input, double scale, double* output, int size)
        self.lib.fast_vector_scale.argtypes = [
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_double,
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_int,
        ]
        self.lib.fast_vector_scale.restype = None

        # void fast_vector_add(const double* a, const double* b, double* output, int size)
        self.lib.fast_vector_add.argtypes = [
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_int,
        ]
        self.lib.fast_vector_add.restype = None

        # double fast_compute_loss(const double* actual, const double* predicted, int size)
        self.lib.fast_compute_loss.argtypes = [
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_int,
        ]
        self.lib.fast_compute_loss.restype = ctypes.c_double

    def dot_product(self, vec_a: List[float], vec_b: List[float]) -> float:
        if len(vec_a) != len(vec_b):
            raise ValueError("Vectors must be of equal length.")
        size = len(vec_a)
        c_a = (ctypes.c_double * size)(*vec_a)
        c_b = (ctypes.c_double * size)(*vec_b)
        return float(self.lib.fast_dot_product(c_a, c_b, size))

    def vector_scale(self, vec: List[float], scale: float) -> List[float]:
        size = len(vec)
        c_in = (ctypes.c_double * size)(*vec)
        c_out = (ctypes.c_double * size)()
        self.lib.fast_vector_scale(c_in, ctypes.c_double(scale), c_out, size)
        return [c_out[i] for i in range(size)]

    def vector_add(self, vec_a: List[float], vec_b: List[float]) -> List[float]:
        if len(vec_a) != len(vec_b):
            raise ValueError("Vectors must be of equal length.")
        size = len(vec_a)
        c_a = (ctypes.c_double * size)(*vec_a)
        c_b = (ctypes.c_double * size)(*vec_b)
        c_out = (ctypes.c_double * size)()
        self.lib.fast_vector_add(c_a, c_b, c_out, size)
        return [c_out[i] for i in range(size)]

    def compute_loss(self, actual: List[float], predicted: List[float]) -> float:
        if len(actual) != len(predicted):
            raise ValueError("Vectors must be of equal length.")
        size = len(actual)
        c_act = (ctypes.c_double * size)(*actual)
        c_pred = (ctypes.c_double * size)(*predicted)
        return float(self.lib.fast_compute_loss(c_act, c_pred, size))
