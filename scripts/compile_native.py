"""Compile C and Rust source code into Windows dynamic libraries (.dll) and shared libraries."""

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
    target_windows: bool = True,
) -> str:
    """Compiles a C source file into a Windows DLL (.dll) and platform shared library.

    Args:
        source_path: Path to .c file.
        output_dir: Target output directory.
        library_name: Output base file name.
        target_windows: Whether to compile for Windows .dll target.

    Returns:
        Path to compiled library.
    """
    os.makedirs(output_dir, exist_ok=True)
    dll_path = os.path.join(output_dir, f"{library_name}.dll")
    so_path = os.path.join(output_dir, f"{library_name}.so")

    # Cross-compile Windows .dll if mingw compiler is available
    if target_windows:
        win_compiler = os.environ.get("WIN_CC", "x86_64-w64-mingw32-gcc")
        try:
            cmd_win = [win_compiler, "-shared", "-O3", "-o", dll_path, source_path]
            subprocess.check_call(cmd_win)
        except Exception as err:
            print(f"Notice: Windows .dll cross-compilation: {err}", file=sys.stderr)

    # Compile host shared library
    compiler = os.environ.get("CC", "gcc")
    system = platform.system()
    out_target = dll_path if system == "Windows" else so_path

    if system == "Windows":
        cmd = [compiler, "-shared", "-O3", "-o", out_target, source_path]
    else:
        cmd = [compiler, "-shared", "-fPIC", "-O3", "-o", out_target, source_path]

    try:
        subprocess.check_call(cmd)
    except (subprocess.CalledProcessError, FileNotFoundError) as err:
        raise RuntimeError(f"Failed to compile C shared library with command {cmd}: {err}") from err

    if os.path.exists(dll_path):
        return os.path.normpath(dll_path)
    return os.path.normpath(so_path)


def compile_all_native_libraries(output_dir: str = "build") -> dict[str, str]:
    """Compiles both C and Rust dynamic libraries with Windows .dll support."""
    c_lib = compile_native_library(output_dir=output_dir, target_windows=True)
    rust_lib = compile_rust_library(output_dir=output_dir, target_windows=True)
    return {"c_library": c_lib, "rust_library": rust_lib}


if __name__ == "__main__":
    res = compile_all_native_libraries()
    print(f"Compiled native C library: {res['c_library']}")
    print(f"Compiled native Rust library: {res['rust_library']}")
