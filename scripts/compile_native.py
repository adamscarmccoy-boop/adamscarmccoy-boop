"""Compile C source code into precompiled shared libraries (.dll for Windows, .so for test environments)."""

from __future__ import annotations

import os
import platform
import subprocess
import sys


def compile_native_library(
    source_path: str = "src/native_bridge.c",
    output_dir: str = "build",
    library_name: str = "native_bridge",
    target_format: str = "auto",
) -> str:
    """Compiles a C source file into dynamic libraries (.dll or .so).

    Args:
        source_path: Path to .c file.
        output_dir: Target output directory.
        library_name: Output base file name.
        target_format: 'dll', 'so', or 'auto'.

    Returns:
        Path to compiled library.
    """
    os.makedirs(output_dir, exist_ok=True)
    system = platform.system()

    if target_format == "dll":
        ext = ".dll"
    elif target_format == "so":
        ext = ".so"
    else:
        ext = ".dll" if system == "Windows" else ".so"

    output_path = os.path.join(output_dir, f"{library_name}{ext}")
    compiler = os.environ.get("CC", "gcc")

    if ext == ".dll":
        cmd = [compiler, "-shared", "-O3", "-o", output_path, source_path]
    else:
        cmd = [compiler, "-shared", "-fPIC", "-O3", "-o", output_path, source_path]

    try:
        subprocess.check_call(cmd)
    except (subprocess.CalledProcessError, FileNotFoundError) as err:
        raise RuntimeError(f"Failed to compile C library with command {cmd}: {err}") from err

    return os.path.normpath(output_path)


def compile_all_targets() -> None:
    """Precompiles both .so (for test execution environment) and .dll (for Windows target environment)."""
    so_path = compile_native_library(target_format="so")
    print(f"Precompiled Linux test shared library: {so_path}")

    try:
        dll_path = compile_native_library(target_format="dll")
        print(f"Precompiled Windows target DLL: {dll_path}")
    except Exception as err:
        print(f"Notice: DLL build on non-Windows compiler fallback: {err}")


if __name__ == "__main__":
    compile_all_targets()
