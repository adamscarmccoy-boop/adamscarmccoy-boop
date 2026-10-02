import json
import os
import shutil
import tempfile
import unittest

from scripts.custom_embed import (
    prepare_documents_from_jsonl,
    format_instruction_for_nv_embed,
    batch_documents,
    generate_embeddings_mock_fallback,
)


class TestCustomEmbed(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_format_instruction_for_nv_embed(self):
        formatted = format_instruction_for_nv_embed("hello world")
        self.assertTrue(formatted.startswith("Instruct:"))
        self.assertTrue(formatted.endswith("hello world"))

    def test_batch_documents(self):
        docs = [{"id": f"doc_{i}"} for i in range(10)]
        batches = batch_documents(docs, batch_size=3)
        self.assertEqual(len(batches), 4)
        self.assertEqual(len(batches[0]), 3)
        self.assertEqual(len(batches[-1]), 1)

    def test_prepare_documents_from_jsonl(self):
        jsonl_file = os.path.join(self.test_dir, "test.jsonl")
        records = [
            {"text": "Simple pretrain text", "metadata": {"file": "a.py"}},
            {"messages": [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"}]},
            {"instruction": "Do x", "input": "y", "output": "z"}
        ]
        with open(jsonl_file, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        docs = prepare_documents_from_jsonl(jsonl_file)
        self.assertEqual(len(docs), 3)
        self.assertEqual(docs[0]["id"], "doc_0")
        self.assertIn("USER: Hi", docs[1]["text"])
        self.assertIn("Instruction: Do x", docs[2]["text"])

    def test_generate_embeddings_fallback(self):
        docs = [{"id": "doc_0", "text": "sample text", "metadata": {"path": "foo"}}]
        manifest = generate_embeddings_mock_fallback(docs, "nvidia/NV-Embed-v2")
        self.assertEqual(manifest["total_documents"], 1)
        self.assertEqual(manifest["model"], "nvidia/NV-Embed-v2")
        self.assertEqual(len(manifest["embeddings"]), 1)


if __name__ == "__main__":
    unittest.main()
