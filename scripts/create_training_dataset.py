#!/usr/bin/env python3
"""Script to extract text files from a directory and format them into LLM training datasets (JSONL)."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Iterator, List, Dict, Any


IGNORE_DIRS = {
    ".git",
    "node_modules",
    "__pycache__",
    ".venv",
    "venv",
    ".idea",
    ".vscode",
    "dist",
    "build",
    ".next",
}

BINARY_EXTENSIONS = {
    ".exe", ".dll", ".so", ".dylib", ".pyc", ".pyd", ".png", ".jpg", ".jpeg",
    ".gif", ".ico", ".svg", ".zip", ".tar", ".gz", ".7z", ".pdf", ".db",
    ".sqlite", ".bin", ".dat", ".wof", ".woff", ".woff2", ".ttf", ".eot",
    ".mp3", ".mp4", ".wav", ".avi", ".mov", ".webm"
}


def is_binary(file_path: str, chunk_size: int = 1024) -> bool:
    """Check if file appears to be binary by looking for null bytes in initial chunk."""
    try:
        with open(file_path, "rb") as f:
            chunk = f.read(chunk_size)
            if b"\0" in chunk:
                return True
            return False
    except Exception:
        return True


def chunk_text(text: str, max_chars: int) -> List[str]:
    """Split text into chunks of maximum max_chars length."""
    if not max_chars or len(text) <= max_chars:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = start + max_chars
        chunks.append(text[start:end])
        start = end
    return chunks


def extract_files(
    root_dir: str,
    allowed_extensions: List[str] | None = None
) -> Iterator[tuple[str, str]]:
    """Recursively yield (relative_path, content) for text files in root_dir."""
    root_path = os.path.abspath(root_dir)

    normalized_extensions = None
    if allowed_extensions:
        normalized_extensions = [
            ext.lower() if ext.startswith(".") else f".{ext.lower()}"
            for ext in allowed_extensions
        ]

    for current_root, dirs, files in os.walk(root_path):
        # Prune ignored directories
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]

        for file_name in files:
            ext = os.path.splitext(file_name)[1].lower()
            if ext in BINARY_EXTENSIONS:
                continue

            if normalized_extensions and ext not in normalized_extensions:
                continue

            full_path = os.path.join(current_root, file_name)
            rel_path = os.path.relpath(full_path, root_path)

            if is_binary(full_path):
                continue

            try:
                with open(full_path, "r", encoding="utf-8", errors="replace") as f:
                    content = f.read()
                    if content.strip():
                        yield rel_path, content
            except Exception as e:
                sys.stderr.write(f"Skipping {rel_path}: {e}\n")


def build_dataset_record(
    rel_path: str,
    content_chunk: str,
    fmt: str
) -> Dict[str, Any]:
    """Build a dictionary record for JSONL output depending on format."""
    if fmt == "text":
        return {
            "text": f"File: {rel_path}\n\n{content_chunk}",
            "metadata": {"path": rel_path}
        }
    elif fmt == "chat":
        return {
            "messages": [
                {
                    "role": "user",
                    "content": f"Provide the contents of file `{rel_path}`."
                },
                {
                    "role": "assistant",
                    "content": content_chunk
                }
            ],
            "metadata": {"path": rel_path}
        }
    elif fmt == "instruction":
        return {
            "instruction": f"Review and reference the code or text in file `{rel_path}`.",
            "input": f"Path: {rel_path}",
            "output": content_chunk,
            "metadata": {"path": rel_path}
        }
    else:
        raise ValueError(f"Unknown format: {fmt}")


def create_dataset(
    input_dir: str,
    output_file: str,
    fmt: str = "text",
    max_chars: int = 4000,
    allowed_extensions: List[str] | None = None
) -> int:
    """Extract files from input_dir and write JSONL dataset to output_file."""
    count = 0
    with open(output_file, "w", encoding="utf-8") as out:
        for rel_path, content in extract_files(input_dir, allowed_extensions):
            chunks = chunk_text(content, max_chars)
            for chunk in chunks:
                record = build_dataset_record(rel_path, chunk, fmt)
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1
    return count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert folder of code/text files into LLM training datasets (JSONL)."
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Path to the directory containing files to extract."
    )
    parser.add_argument(
        "--output-file",
        required=True,
        help="Path to output .jsonl dataset file."
    )
    parser.add_argument(
        "--format",
        choices=["text", "chat", "instruction"],
        default="text",
        help="Dataset output format: 'text' (pretraining), 'chat' (SFT messages), or 'instruction' (Alpaca style)."
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=4000,
        help="Maximum characters per chunk per dataset record."
    )
    parser.add_argument(
        "--extensions",
        nargs="*",
        help="Optional list of file extensions to include (e.g. py json md ts or .py .json)."
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    print(f"Extracting dataset from: {args.input_dir}")
    count = create_dataset(
        input_dir=args.input_dir,
        output_file=args.output_file,
        fmt=args.format,
        max_chars=args.max_chars,
        allowed_extensions=args.extensions
    )
    print(f"Successfully wrote {count} records to {args.output_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
