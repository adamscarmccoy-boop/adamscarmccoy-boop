#!/usr/bin/env python3
"""Generate embeddings via LM Studio local server (port 1234) using Nemotron models."""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from typing import List, Dict, Any


DEFAULT_BASE_URL = "http://127.0.0.1:1234/v1"
DEFAULT_MODEL = "text-embedding-llama-nemotron-embed-1b-v2"


def prepare_texts_from_jsonl(jsonl_path: str) -> List[Dict[str, Any]]:
    """Extract document texts and metadata from JSONL dataset."""
    if not os.path.exists(jsonl_path):
        raise FileNotFoundError(f"Input JSONL file not found at {jsonl_path}")

    records = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)

            text = ""
            if "text" in item:
                text = item["text"]
            elif "messages" in item:
                text = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in item["messages"]])
            elif "instruction" in item:
                text = f"{item['instruction']}\n{item.get('input', '')}\n{item.get('output', '')}"

            if text.strip():
                records.append({
                    "id": f"doc_{idx}",
                    "text": text,
                    "metadata": item.get("metadata", {})
                })
    return records


def send_embedding_request(
    base_url: str,
    api_key: str,
    model: str,
    input_texts: List[str]
) -> List[List[float]]:
    """Send /v1/embeddings POST request to LM Studio server."""
    endpoint = base_url.rstrip("/") + "/embeddings"
    payload = {
        "model": model,
        "input": input_texts
    }

    headers = {
        "Content-Type": "application/json",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    req = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST"
    )

    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            # Standard OpenAI embeddings format: data["data"][i]["embedding"]
            return [item["embedding"] for item in data.get("data", [])]
    except urllib.error.HTTPError as err:
        sys.stderr.write(f"HTTP Error: {err.read().decode('utf-8')}\n")
        raise
    except Exception as err:
        sys.stderr.write(f"Connection failed to LM Studio at {endpoint}: {err}\n")
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate vector embeddings via LM Studio (port 1234) with Nemotron models."
    )
    parser.add_argument("--jsonl-path", required=True, help="Input JSONL file.")
    parser.add_argument("--output-path", default="lm_studio_embeddings.json", help="Output JSON path.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help=f"LM Studio base URL (default: {DEFAULT_BASE_URL}).")
    parser.add_argument("--api-key", default="lm-studio", help="API Key for LM Studio.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Embedding model name (default: {DEFAULT_MODEL}).")
    parser.add_argument("--batch-size", type=int, default=8, help="Texts per embedding request batch.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    print(f"Reading dataset: {args.jsonl_path}")
    docs = prepare_texts_from_jsonl(args.jsonl_path)
    print(f"Loaded {len(docs)} documents.")

    all_embeddings = []
    for i in range(0, len(docs), args.batch_size):
        batch = docs[i : i + args.batch_size]
        texts = [d["text"] for d in batch]
        print(f"Embedding batch {i // args.batch_size + 1} ({len(texts)} texts) using {args.model} at {args.base_url}...")

        try:
            embeddings = send_embedding_request(
                base_url=args.base_url,
                api_key=args.api_key,
                model=args.model,
                input_texts=texts
            )
            for doc, emb in zip(batch, embeddings):
                all_embeddings.append({
                    "id": doc["id"],
                    "metadata": doc["metadata"],
                    "embedding_dim": len(emb),
                    "embedding": emb
                })
        except Exception as e:
            sys.stderr.write(f"Batch failed: {e}\n")
            return 1

    manifest = {
        "model": args.model,
        "base_url": args.base_url,
        "total_documents": len(all_embeddings),
        "data": all_embeddings
    }

    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Successfully saved LM Studio embeddings manifest to {args.output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
