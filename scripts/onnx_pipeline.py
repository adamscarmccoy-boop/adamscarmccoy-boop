#!/usr/bin/env python3
"""End-to-end PyTorch to ONNX model export, ONNX Runtime execution, and English text decoding."""

from __future__ import annotations

import argparse
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import onnx
import onnxruntime as ort

# Character vocabulary mapping for ASCII English decoding
VOCAB = [chr(i) for i in range(32, 127)]  # Printable ASCII
CHAR_TO_ID = {c: i for i, c in enumerate(VOCAB)}
ID_TO_CHAR = {i: c for i, c in enumerate(VOCAB)}
VOCAB_SIZE = len(VOCAB)


class TextDecoderModel(nn.Module):
    """Neural model projecting text embedding vectors to vocabulary logits."""
    def __init__(self, embed_dim: int = 128, vocab_size: int = VOCAB_SIZE):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.GELU(),
            nn.Linear(256, vocab_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


def get_character_embedding(char_id: int, embed_dim: int = 128) -> np.ndarray:
    """Generate embedding vector for a character ID."""
    vec = np.zeros(embed_dim, dtype=np.float32)
    for d in range(embed_dim):
        vec[d] = np.sin((char_id + 1) * 0.1 + d * 0.05)
    return vec


def encode_english_text(text: str, embed_dim: int = 128) -> np.ndarray:
    """Encode English text into character embedding vectors."""
    vectors = []
    for char in text:
        cid = CHAR_TO_ID.get(char, 0)
        vec = get_character_embedding(cid, embed_dim)
        vectors.append(vec)
    return np.array(vectors, dtype=np.float32)


def decode_logits_to_english(logits: np.ndarray) -> str:
    """Argmax logit decoding back to English string."""
    token_ids = np.argmax(logits, axis=-1)
    chars = [ID_TO_CHAR.get(tid, "?") for tid in token_ids]
    return "".join(chars)


def export_model_to_onnx(onnx_path: str, embed_dim: int = 128) -> None:
    """Instantiate model, train weights on ASCII vocabulary, and export to ONNX."""
    torch.manual_seed(42)
    model = TextDecoderModel(embed_dim=embed_dim, vocab_size=VOCAB_SIZE)

    # Train model to map character embeddings accurately to vocabulary logits
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    X_train = []
    y_train = []
    for cid in range(VOCAB_SIZE):
        X_train.append(get_character_embedding(cid, embed_dim))
        y_train.append(cid)

    X_tensor = torch.tensor(np.array(X_train), dtype=torch.float32)
    y_tensor = torch.tensor(np.array(y_train), dtype=torch.int64)

    model.train()
    for _ in range(500):
        optimizer.zero_grad()
        logits = model(X_tensor)
        loss = criterion(logits, y_tensor)
        loss.backward()
        optimizer.step()

    model.eval()

    dummy_input = torch.randn(1, embed_dim, dtype=torch.float32)

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=["input_embeddings"],
        output_names=["vocab_logits"],
        dynamo=False,
        dynamic_axes={
            "input_embeddings": {0: "batch_size"},
            "vocab_logits": {0: "batch_size"}
        }
    )
    print(f"Successfully exported ONNX model to {onnx_path}")


def run_onnx_inference(onnx_path: str, input_vectors: np.ndarray) -> np.ndarray:
    """Run ONNX Runtime inference using exported .onnx model file."""
    session = ort.InferenceSession(onnx_path)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    logits = session.run([output_name], {input_name: input_vectors})[0]
    return logits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="End-to-end PyTorch to ONNX export, execution, and English decoding."
    )
    parser.add_argument("--onnx-path", default="text_decoder.onnx", help="Output path for .onnx file.")
    parser.add_argument("--input-text", default="Hello World! Nemotron ONNX Pipeline Ready.", help="English text to encode/decode.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    print(f"1. Training & Exporting PyTorch model to ONNX ({args.onnx_path})...")
    export_model_to_onnx(args.onnx_path)

    print(f"\n2. Encoding English input text: '{args.input_text}'")
    vectors = encode_english_text(args.input_text)

    print(f"\n3. Running ONNX Runtime inference over {len(vectors)} token vectors...")
    logits = run_onnx_inference(args.onnx_path, vectors)

    print("\n4. Decoding ONNX logits back to English text...")
    decoded_text = decode_logits_to_english(logits)
    print(f"Decoded Output Text: '{decoded_text}'")

    if decoded_text == args.input_text:
        print("\nSUCCESS: End-to-end ONNX inference & English decoding verified 100%!")
        return 0
    else:
        print(f"\nMismatch: '{decoded_text}' != '{args.input_text}'")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
