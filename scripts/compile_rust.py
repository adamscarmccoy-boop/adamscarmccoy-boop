"""Compile Rust source code into shared libraries (.so for Linux / .dll for Windows)."""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys


def compile_rust_library(
    crate_dir: str = "src/rust_arrow_bridge",
    output_dir: str = "build",
    library_name: str = "rust_arrow_bridge",
) -> str:
    """Compiles a Rust crate into a platform dynamic library (.so or .dll).

    Args:
        crate_dir: Path to Rust crate directory containing Cargo.toml.
        output_dir: Target output directory.
        library_name: Output base file name.

    Returns:
        Path to compiled library in output_dir.
    """
    os.makedirs(output_dir, exist_ok=True)
    system = platform.system()

    cmd = ["cargo", "build", "--release", "--manifest-path", os.path.join(crate_dir, "Cargo.toml")]

    try:
        subprocess.check_call(cmd)
    except (subprocess.CalledProcessError, FileNotFoundError) as err:
        raise RuntimeError(f"Failed to compile Rust shared library with command {cmd}: {err}") from err

    target_release = os.path.join(crate_dir, "target", "release")

    if system == "Windows":
        source_lib = os.path.join(target_release, f"{library_name}.dll")
        dest_lib = os.path.join(output_dir, f"{library_name}.dll")
    elif system == "Darwin":
        source_lib = os.path.join(target_release, f"lib{library_name}.dylib")
        dest_lib = os.path.join(output_dir, f"{library_name}.dylib")
    else:
        source_lib = os.path.join(target_release, f"lib{library_name}.so")
        dest_lib = os.path.join(output_dir, f"{library_name}.so")

    if not os.path.exists(source_lib):
        # Fallback check for non-lib prefix or alternative output
        alt_source = os.path.join(target_release, f"{library_name}.so")
        if os.path.exists(alt_source):
            source_lib = alt_source

    if not os.path.exists(source_lib):
        raise FileNotFoundError(f"Compiled Rust artifact not found at {source_lib}")

    shutil.copy2(source_lib, dest_lib)
    return os.path.normpath(dest_lib)


if __name__ == "__main__":
    out = compile_rust_library()
    print(f"Compiled Rust native library to: {out}")
