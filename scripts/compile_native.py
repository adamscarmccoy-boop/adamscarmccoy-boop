"""Compile C source code into Windows dynamic libraries (.dll)."""

from __future__ import annotations

import os
import subprocess
import sys


def compile_native_library(
    source_path: str = "src/native_bridge.c",
    output_dir: str = "build",
    library_name: str = "native_bridge",
) -> str:
    """Compiles a C source file into a Windows DLL (.dll).

    Args:
        source_path: Path to .c file.
        output_dir: Target output directory.
        library_name: Output base file name.

    Returns:
        Path to compiled DLL library.
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{library_name}.dll")

    compiler = os.environ.get("CC", "gcc")
    cmd = [compiler, "-shared", "-O3", "-o", output_path, source_path]

    try:
        subprocess.check_call(cmd)
    except (subprocess.CalledProcessError, FileNotFoundError) as err:
        raise RuntimeError(f"Failed to compile C DLL with command {cmd}: {err}") from err

    return os.path.normpath(output_path)


if __name__ == "__main__":
    out = compile_native_library()
    print(f"Compiled native Windows DLL to: {out}")
