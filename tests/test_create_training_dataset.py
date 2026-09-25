import json
import os
import shutil
import tempfile
import unittest

from scripts.create_training_dataset import (
    chunk_text,
    is_binary,
    extract_files,
    build_dataset_record,
    create_dataset,
)


class TestCreateTrainingDataset(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_chunk_text(self):
        text = "abcdefghij"
        chunks = chunk_text(text, max_chars=4)
        self.assertEqual(chunks, ["abcd", "efgh", "ij"])

    def test_is_binary(self):
        text_file = os.path.join(self.test_dir, "test.txt")
        with open(text_file, "w") as f:
            f.write("Hello World")

        bin_file = os.path.join(self.test_dir, "test.bin")
        with open(bin_file, "wb") as f:
            f.write(b"Hello\0World")

        self.assertFalse(is_binary(text_file))
        self.assertTrue(is_binary(bin_file))

    def test_extract_files_and_create_dataset(self):
        os.makedirs(os.path.join(self.test_dir, "subfolder"))
        file1 = os.path.join(self.test_dir, "file1.py")
        file2 = os.path.join(self.test_dir, "subfolder", "file2.md")

        with open(file1, "w") as f:
            f.write("print('hello')\n")
        with open(file2, "w") as f:
            f.write("# Title\nContent\n")

        output_jsonl = os.path.join(self.test_dir, "dataset.jsonl")
        count = create_dataset(
            input_dir=self.test_dir,
            output_file=output_jsonl,
            fmt="chat",
            max_chars=100
        )

        self.assertEqual(count, 2)
        self.assertTrue(os.path.exists(output_jsonl))

        with open(output_jsonl, "r") as f:
            lines = f.readlines()
            self.assertEqual(len(lines), 2)
            data0 = json.loads(lines[0])
            self.assertIn("messages", data0)
            self.assertIn("metadata", data0)

    def test_build_dataset_record_formats(self):
        rec_text = build_dataset_record("foo.py", "code", "text")
        self.assertIn("text", rec_text)

        rec_chat = build_dataset_record("foo.py", "code", "chat")
        self.assertIn("messages", rec_chat)

        rec_inst = build_dataset_record("foo.py", "code", "instruction")
        self.assertIn("instruction", rec_inst)


if __name__ == "__main__":
    unittest.main()
