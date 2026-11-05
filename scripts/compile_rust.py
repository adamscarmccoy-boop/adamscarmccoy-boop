"""Compile Rust source code into Windows dynamic link libraries (.dll) and platform shared libraries."""

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
    target_windows: bool = True,
) -> str:
    """Compiles a Rust crate into a Windows dynamic library (.dll) and platform shared object.

    Args:
        crate_dir: Path to Rust crate directory containing Cargo.toml.
        output_dir: Target output directory.
        library_name: Output base file name.
        target_windows: Whether to build for Windows x86_64 target (.dll).

    Returns:
        Path to compiled library in output_dir.
    """
    os.makedirs(output_dir, exist_ok=True)
    manifest_path = os.path.join(crate_dir, "Cargo.toml")

    # Always compile Windows .dll if mingw / cross-target is available
    if target_windows:
        win_target = "x86_64-pc-windows-gnu"
        win_cmd = [
            "cargo",
            "build",
            "--release",
            "--target",
            win_target,
            "--manifest-path",
            manifest_path,
        ]
        env = os.environ.copy()
        env["PYO3_CROSS_PYTHON_VERSION"] = "3.12"
        if os.path.exists("/tmp/mingw_lib"):
            env["RUSTFLAGS"] = "-L /tmp/mingw_lib"

        try:
            subprocess.check_call(win_cmd, env=env)
            win_src = os.path.join(
                crate_dir, "target", win_target, "release", f"{library_name}.dll"
            )
            win_dest = os.path.join(output_dir, f"{library_name}.dll")
            if os.path.exists(win_src):
                shutil.copy2(win_src, win_dest)
        except Exception as err:
            print(f"Windows target compilation notice: {err}", file=sys.stderr)

    # Also build host target (.so for Linux CI execution)
    host_cmd = ["cargo", "build", "--release", "--manifest-path", manifest_path]
    try:
        subprocess.check_call(host_cmd)
    except (subprocess.CalledProcessError, FileNotFoundError) as err:
        raise RuntimeError(f"Failed to compile Rust library: {err}") from err

    host_release = os.path.join(crate_dir, "target", "release")
    system = platform.system()
    if system == "Windows":
        source_lib = os.path.join(host_release, f"{library_name}.dll")
        dest_lib = os.path.join(output_dir, f"{library_name}.dll")
    elif system == "Darwin":
        source_lib = os.path.join(host_release, f"lib{library_name}.dylib")
        dest_lib = os.path.join(output_dir, f"{library_name}.dylib")
    else:
        source_lib = os.path.join(host_release, f"lib{library_name}.so")
        dest_lib = os.path.join(output_dir, f"{library_name}.so")

    if not os.path.exists(source_lib):
        alt_source = os.path.join(host_release, f"{library_name}.so")
        if os.path.exists(alt_source):
            source_lib = alt_source

    if os.path.exists(source_lib):
        shutil.copy2(source_lib, dest_lib)

    # Return Windows .dll if present, otherwise dest_lib
    win_final = os.path.join(output_dir, f"{library_name}.dll")
    if os.path.exists(win_final):
        return os.path.normpath(win_final)
    return os.path.normpath(dest_lib)


if __name__ == "__main__":
    out = compile_rust_library()
    print(f"Compiled Rust native library to: {out}")
