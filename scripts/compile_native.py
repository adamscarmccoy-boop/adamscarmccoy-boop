"""Compile C and Rust source code into shared libraries (.so for Linux / .dll for Windows)."""

from __future__ import annotations

import os
import platform
import subprocess
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.compile_rust import compile_rust_library


def compile_native_library(
    source_path: str = "src/native_bridge.c",
    output_dir: str = "build",
    library_name: str = "native_bridge",
) -> str:
    """Compiles a C source file into a platform dynamic library (.so or .dll).

    Args:
        source_path: Path to .c file.
        output_dir: Target output directory.
        library_name: Output base file name.

    Returns:
        Path to compiled library.
    """
    os.makedirs(output_dir, exist_ok=True)
    system = platform.system()

    if system == "Windows":
        output_path = os.path.join(output_dir, f"{library_name}.dll")
    else:
        output_path = os.path.join(output_dir, f"{library_name}.so")

    compiler = os.environ.get("CC", "gcc")

    if system == "Windows":
        cmd = [compiler, "-shared", "-O3", "-o", output_path, source_path]
    else:
        cmd = [compiler, "-shared", "-fPIC", "-O3", "-o", output_path, source_path]

    try:
        subprocess.check_call(cmd)
    except (subprocess.CalledProcessError, FileNotFoundError) as err:
        raise RuntimeError(f"Failed to compile C shared library with command {cmd}: {err}") from err

    return os.path.normpath(output_path)


def compile_all_native_libraries(output_dir: str = "build") -> dict[str, str]:
    """Compiles both C and Rust dynamic shared object libraries."""
    c_lib = compile_native_library(output_dir=output_dir)
    rust_lib = compile_rust_library(output_dir=output_dir)
    return {"c_library": c_lib, "rust_library": rust_lib}


if __name__ == "__main__":
    res = compile_all_native_libraries()
    print(f"Compiled native C library to: {res['c_library']}")
    print(f"Compiled native Rust library to: {res['rust_library']}")
