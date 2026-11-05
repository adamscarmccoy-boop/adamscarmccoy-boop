"""Python PyArrow, Polars, DuckDB, LanceDB, and Memory Mapping (mmap) bridge runner interfacing with Rust dynamic link libraries."""

from __future__ import annotations

import ctypes
import mmap
import os
import platform
import struct
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pyarrow as pa

try:
    import polars as pl
except ImportError:
    pl = None

try:
    import duckdb
except ImportError:
    duckdb = None

try:
    import lancedb
except ImportError:
    lancedb = None

from scripts.compile_rust import compile_rust_library


class RustArrowBridgeRunner:
    """Interfaces PyArrow, Polars, DuckDB, LanceDB, and mmap buffers with compiled Rust dynamic libraries (.dll / .so)."""

    def __init__(self, lib_path: Optional[str] = None, prefer_windows_dll: bool = False):
        if lib_path is None:
            system = platform.system()
            # If on Windows or if prefer_windows_dll is requested and available
            win_dll = os.path.join("build", "rust_arrow_bridge.dll")
            unix_so = os.path.join("build", "rust_arrow_bridge.so")

            if system == "Windows":
                default_path = win_dll
            elif prefer_windows_dll and os.path.exists(win_dll):
                default_path = win_dll
            elif os.path.exists(unix_so):
                default_path = unix_so
            elif os.path.exists(win_dll):
                default_path = win_dll
            else:
                default_path = compile_rust_library()
            lib_path = default_path

        self.lib_path = os.path.normpath(lib_path)
        if not os.path.exists(self.lib_path):
            raise FileNotFoundError(f"Rust dynamic library not found at: {self.lib_path}")

        # On non-Windows platforms attempting to load .dll directly may fail ctypes load,
        # so fallback to .so for runtime execution while preserving .dll for Windows deployment
        try:
            self.lib = ctypes.CDLL(self.lib_path)
        except OSError:
            so_alt = os.path.join(os.path.dirname(self.lib_path), "rust_arrow_bridge.so")
            if os.path.exists(so_alt):
                self.lib = ctypes.CDLL(so_alt)
            else:
                raise

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

    def score_table_vectors_aot(
        self,
        table: pa.Table,
        query_vector: List[float],
        vector_column: str = "vector",
        score_column: str = "rust_aot_score",
    ) -> pa.Table:
        """In-table model execution: Scores each vector row against a query vector using Rust AOT dot product."""
        if vector_column not in table.column_names:
            raise KeyError(f"Vector column '{vector_column}' not found in Arrow Table.")

        vectors = table.column(vector_column).to_pylist()
        scores = [self.dot_product(query_vector, row_vec) for row_vec in vectors]
        new_col = pa.array(scores, type=pa.float64())
        return table.append_column(score_column, new_col)

    @staticmethod
    def load_to_duckdb(
        table: pa.Table,
        table_name: str = "arrow_table",
        connection: Optional[Any] = None,
    ) -> Any:
        """Loads and registers an Arrow Table zero-copy into DuckDB."""
        if duckdb is None:
            raise ImportError("DuckDB is not installed.")
        con = connection or duckdb.connect()
        con.register(table_name, table)
        return con

    @staticmethod
    def load_to_lancedb(
        table: pa.Table,
        db_path: str = "/tmp/lancedb_repo",
        table_name: str = "vectors",
        mode: str = "overwrite",
    ) -> Any:
        """Loads and stores an Arrow Table with vector embeddings into LanceDB."""
        if lancedb is None:
            raise ImportError("LanceDB is not installed.")
        db = lancedb.connect(db_path)
        return db.create_table(table_name, data=table, mode=mode)

    @staticmethod
    def create_mmap_808_file(
        filepath: str,
        num_records: int = 5,
        dim_f64: int = 101,  # 101 * 8 bytes = 808 bytes per vector record
    ) -> str:
        """Creates a binary memory-mapped file containing 808-byte aligned vector records (101 x float64)."""
        bytes_per_record = dim_f64 * 8
        assert bytes_per_record == 808, f"Record size must be exactly 808 bytes, got {bytes_per_record}"

        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)
        with open(filepath, "wb") as f:
            for rec_idx in range(num_records):
                # Deterministic float values for testing
                floats = [float(rec_idx + i * 0.01) for i in range(dim_f64)]
                buf = struct.pack(f"{dim_f64}d", *floats)
                assert len(buf) == 808
                f.write(buf)

        return os.path.normpath(filepath)

    @classmethod
    def read_mmap_808_table(
        cls,
        filepath: str,
        num_records: int = 5,
        dim_f64: int = 101,
    ) -> Tuple[pa.Table, mmap.mmap]:
        """Memory-maps an 808-byte record aligned file and converts to Apache Arrow Table zero-copy."""
        bytes_per_record = dim_f64 * 8
        assert bytes_per_record == 808

        f = open(filepath, "r+b")
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)

        record_vectors = []
        raw_byte_chunks = []
        for i in range(num_records):
            offset = i * 808
            chunk = mm[offset : offset + 808]
            raw_byte_chunks.append(chunk)
            floats = list(struct.unpack(f"{dim_f64}d", chunk))
            record_vectors.append(floats)

        table = pa.Table.from_pydict({
            "id": list(range(1, num_records + 1)),
            "vector": record_vectors,
            "raw_808_bytes": raw_byte_chunks,
        })
        return table, mm


if __name__ == "__main__":
    runner = RustArrowBridgeRunner()
    arr = pa.array([1.0, 2.0, 3.0, 4.0])
    scaled = runner.vector_scale(arr, 2.0)
    print(f"Original Arrow Array: {arr}")
    print(f"Scaled by Rust: {scaled}")
