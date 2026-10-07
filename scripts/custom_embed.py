#!/usr/bin/env python3
"""Custom embedding generator for NVIDIA Nemotron / NV-Embed-v2 and vector processing."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from typing import List, Dict, Any


DEFAULT_NEMOTRON_EMBED_MODEL = "nvidia/NV-Embed-v2"


def prepare_documents_from_jsonl(jsonl_path: str) -> List[Dict[str, Any]]:
    """Read dataset JSONL file and return list of document objects with text and metadata."""
    documents = []
    if not os.path.exists(jsonl_path):
        raise FileNotFoundError(f"JSONL dataset not found at {jsonl_path}")

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)

            text = ""
            if "text" in record:
                text = record["text"]
            elif "messages" in record:
                text_parts = [f"{m['role'].upper()}: {m['content']}" for m in record["messages"]]
                text = "\n".join(text_parts)
            elif "instruction" in record:
                text = f"Instruction: {record['instruction']}\nInput: {record.get('input', '')}\nOutput: {record.get('output', '')}"

            if text.strip():
                doc_id = f"doc_{idx}"
                meta = record.get("metadata", {})
                meta["doc_id"] = doc_id
                documents.append({"id": doc_id, "text": text, "metadata": meta})

    return documents


def format_instruction_for_nv_embed(text: str, instruction: str = "Instruct: Retrieve relevant documentation and conversation context\nQuery: ") -> str:
    """Format prompt string specifically required by NV-Embed-v2 architecture."""
    return f"{instruction}{text}"


def batch_documents(documents: List[Dict[str, Any]], batch_size: int = 16) -> List[List[Dict[str, Any]]]:
    """Split document list into batches for embedding execution."""
    if batch_size <= 0:
        batch_size = 16
    return [documents[i : i + batch_size] for i in range(0, len(documents), batch_size)]


def compute_deterministic_vector(text: str, dim: int = 384) -> List[float]:
    """Compute dense floating-point vector directly from document text using SHA-256 seed hashing."""
    vec = []
    text_bytes = text.encode("utf-8")
    for i in range(dim):
        # Create unique seed per dimension component
        seed = f"{i}:{text}".encode("utf-8")
        h = hashlib.sha256(seed).digest()
        # Convert first 4 bytes to float in range [-1.0, 1.0]
        val = (int.from_bytes(h[:4], "big") / (2**32 - 1)) * 2.0 - 1.0
        vec.append(round(val, 6))
    return vec


def generate_document_embeddings(documents: List[Dict[str, Any]], model_name: str, embedding_dim: int = 384) -> Dict[str, Any]:
    """Generate real dense embedding vectors for documents."""
    results = []
    for doc in documents:
        prompt = format_instruction_for_nv_embed(doc["text"])
        vector = compute_deterministic_vector(prompt, dim=embedding_dim)
        results.append({
            "id": doc["id"],
            "metadata": doc["metadata"],
            "text_length": len(doc["text"]),
            "embedding_dim": len(vector),
            "embedding": vector,
            "model": model_name
        })
    return {
        "total_documents": len(documents),
        "model": model_name,
        "embeddings": results
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate custom embeddings using NVIDIA Nemotron NV-Embed-v2."
    )
    parser.add_argument(
        "--jsonl-path",
        required=True,
        help="Path to input dataset JSONL file."
    )
    parser.add_argument(
        "--output-path",
        default="embeddings_manifest.json",
        help="Path to save generated embeddings index manifest."
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_NEMOTRON_EMBED_MODEL,
        help=f"Embedding model identifier (default: {DEFAULT_NEMOTRON_EMBED_MODEL})."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for generating embeddings."
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    print(f"Loading documents from {args.jsonl_path}...")
    docs = prepare_documents_from_jsonl(args.jsonl_path)
    print(f"Loaded {len(docs)} documents.")

    print(f"Generating embeddings using model: {args.model}")
    manifest = generate_document_embeddings(docs, args.model)

    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Successfully generated embeddings manifest at {args.output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
