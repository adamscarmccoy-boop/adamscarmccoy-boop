import os
import shutil
import tempfile
import unittest
import numpy as np

from scripts.onnx_pipeline import (
    encode_english_text,
    decode_logits_to_english,
    export_model_to_onnx,
    run_onnx_inference,
)


class TestONNXPipeline(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.onnx_path = os.path.join(self.test_dir, "test_decoder.onnx")

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_encode_and_export_onnx(self):
        export_model_to_onnx(self.onnx_path, embed_dim=128)
        self.assertTrue(os.path.exists(self.onnx_path))

        test_string = "Hello ONNX!"
        vectors = encode_english_text(test_string, embed_dim=128)
        self.assertEqual(len(vectors), len(test_string))

        logits = run_onnx_inference(self.onnx_path, vectors)
        decoded = decode_logits_to_english(logits)
        self.assertEqual(decoded, test_string)


if __name__ == "__main__":
    unittest.main()
