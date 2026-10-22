"""Python PyArrow and Polars bridge runner interfacing with Rust dynamic shared objects."""

from __future__ import annotations

import ctypes
import os
import platform
from typing import Any, List, Optional, Union

import numpy as np
import pyarrow as pa

try:
    import polars as pl
except ImportError:
    pl = None

from scripts.compile_rust import compile_rust_library


class RustArrowBridgeRunner:
    """Interfaces PyArrow and Polars data structures with compiled Rust shared objects."""

    def __init__(self, lib_path: Optional[str] = None):
        if lib_path is None:
            system = platform.system()
            ext = ".dll" if system == "Windows" else ".so"
            default_path = os.path.join("build", f"rust_arrow_bridge{ext}")
            if not os.path.exists(default_path):
                default_path = compile_rust_library()
            lib_path = default_path

        self.lib_path = os.path.normpath(lib_path)
        if not os.path.exists(self.lib_path):
            raise FileNotFoundError(f"Rust native library not found at: {self.lib_path}")

        self.lib = ctypes.CDLL(self.lib_path)
        self._setup_function_signatures()

    def _setup_function_signatures(self) -> None:
        # double rust_fast_dot_product(const double* a, const double* b, size_t size)
        self.lib.rust_fast_dot_product.argtypes = [
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_size_t,
        ]
        self.lib.rust_fast_dot_product.restype = ctypes.c_double

        # void rust_fast_vector_scale(const double* input, double scale, double* output, size_t size)
        self.lib.rust_fast_vector_scale.argtypes = [
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_double,
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_size_t,
        ]
        self.lib.rust_fast_vector_scale.restype = None

        # void rust_fast_vector_add(const double* a, const double* b, double* output, size_t size)
        self.lib.rust_fast_vector_add.argtypes = [
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_size_t,
        ]
        self.lib.rust_fast_vector_add.restype = None

        # double rust_fast_compute_loss(const double* actual, const double* predicted, size_t size)
        self.lib.rust_fast_compute_loss.argtypes = [
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_size_t,
        ]
        self.lib.rust_fast_compute_loss.restype = ctypes.c_double

    @staticmethod
    def _extract_float_list(
        data: Union[List[float], pa.Array, pa.ChunkedArray, np.ndarray, Any]
    ) -> List[float]:
        if isinstance(data, list):
            return [float(x) for x in data]
        elif isinstance(data, np.ndarray):
            return data.astype(np.float64).flatten().tolist()
        elif isinstance(data, (pa.Array, pa.ChunkedArray)):
            return data.to_pylist()
        elif pl is not None and isinstance(data, pl.Series):
            return data.to_list()
        else:
            try:
                return [float(x) for x in data]
            except (TypeError, ValueError) as err:
                raise TypeError(f"Unsupported Arrow input data type: {type(data)}") from err

    def dot_product(
        self,
        vec_a: Union[List[float], pa.Array, Any],
        vec_b: Union[List[float], pa.Array, Any],
    ) -> float:
        list_a = self._extract_float_list(vec_a)
        list_b = self._extract_float_list(vec_b)

        if len(list_a) != len(list_b):
            raise ValueError(f"Vector lengths must match ({len(list_a)} vs {len(list_b)})")

        size = len(list_a)
        c_a = (ctypes.c_double * size)(*list_a)
        c_b = (ctypes.c_double * size)(*list_b)
        return float(self.lib.rust_fast_dot_product(c_a, c_b, size))

    def vector_scale(
        self,
        data: Union[List[float], pa.Array, Any],
        scale: float = 1.0,
    ) -> List[float]:
        list_data = self._extract_float_list(data)
        size = len(list_data)
        c_in = (ctypes.c_double * size)(*list_data)
        c_out = (ctypes.c_double * size)()
        self.lib.rust_fast_vector_scale(c_in, ctypes.c_double(scale), c_out, size)
        return [c_out[i] for i in range(size)]

    def vector_add(
        self,
        vec_a: Union[List[float], pa.Array, Any],
        vec_b: Union[List[float], pa.Array, Any],
    ) -> List[float]:
        list_a = self._extract_float_list(vec_a)
        list_b = self._extract_float_list(vec_b)

        if len(list_a) != len(list_b):
            raise ValueError(f"Vector lengths must match ({len(list_a)} vs {len(list_b)})")

        size = len(list_a)
        c_a = (ctypes.c_double * size)(*list_a)
        c_b = (ctypes.c_double * size)(*list_b)
        c_out = (ctypes.c_double * size)()
        self.lib.rust_fast_vector_add(c_a, c_b, c_out, size)
        return [c_out[i] for i in range(size)]

    def compute_loss(
        self,
        actual: Union[List[float], pa.Array, Any],
        predicted: Union[List[float], pa.Array, Any],
    ) -> float:
        list_act = self._extract_float_list(actual)
        list_pred = self._extract_float_list(predicted)

        if len(list_act) != len(list_pred):
            raise ValueError(f"Vector lengths must match ({len(list_act)} vs {len(list_pred)})")

        size = len(list_act)
        c_act = (ctypes.c_double * size)(*list_act)
        c_pred = (ctypes.c_double * size)(*list_pred)
        return float(self.lib.rust_fast_compute_loss(c_act, c_pred, size))

    def prepare_onnx_tensor(
        self,
        data: Union[List[float], pa.Array, Any],
    ) -> np.ndarray:
        """Converts Arrow data into a contiguous float32 tensor array for ONNX Runtime."""
        scaled_data = self.vector_scale(data, scale=1.0)
        return np.array(scaled_data, dtype=np.float32).reshape(1, -1)

    def process_arrow_table(
        self,
        table: pa.Table,
        column_name: str,
        scale: float = 1.5,
        output_col_name: str = "scaled_features",
    ) -> pa.Table:
        """Processes an Apache Arrow Table column through Rust vector operations."""
        if column_name not in table.column_names:
            raise KeyError(f"Column '{column_name}' not found in Arrow Table.")

        col_data = table.column(column_name)
        scaled_res = self.vector_scale(col_data, scale=scale)
        new_col = pa.array(scaled_res, type=pa.float64())

        return table.append_column(output_col_name, new_col)


if __name__ == "__main__":
    runner = RustArrowBridgeRunner()
    arr = pa.array([1.0, 2.0, 3.0, 4.0])
    scaled = runner.vector_scale(arr, 2.0)
    print(f"Original Arrow Array: {arr}")
    print(f"Scaled by Rust: {scaled}")
