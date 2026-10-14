"""Compile C source code into shared libraries (.so for Linux / .dll for Windows)."""

from __future__ import annotations

import os
import platform
import subprocess
import sys


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


if __name__ == "__main__":
    out = compile_native_library()
    print(f"Compiled native library to: {out}")
