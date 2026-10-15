"""Tests for scripts/onnx_engine.py."""

import os
import tempfile
import unittest
import numpy as np

from scripts.onnx_engine import build_linear_model, inspect_onnx_model, run_onnx_inference


class TestOnnxEngine(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.model_path = os.path.join(self.temp_dir.name, "test_model.onnx")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_build_and_inspect_model(self):
        saved_path = build_linear_model(self.model_path, input_dim=3, weights=[1.0, 2.0, 3.0], bias=[0.1, 0.2, 0.3])
        self.assertTrue(os.path.exists(saved_path))

        info = inspect_onnx_model(saved_path)
        self.assertIn("X", info["inputs"])
        self.assertIn("Y", info["outputs"])
        self.assertEqual(info["nodes"], ["Mul", "Add"])

    def test_run_onnx_inference(self):
        build_linear_model(self.model_path, input_dim=4, weights=[2.0, 2.0, 2.0, 2.0], bias=[1.0, 1.0, 1.0, 1.0])
        input_data = [1.0, 2.0, 3.0, 4.0]
        output = run_onnx_inference(self.model_path, input_data)

        expected = np.array([[3.0, 5.0, 7.0, 9.0]], dtype=np.float32)
        np.testing.assert_allclose(output, expected)

    def test_invalid_weights_bias_dim(self):
        with self.assertRaises(ValueError):
            build_linear_model(self.model_path, input_dim=4, weights=[1.0, 2.0])


if __name__ == "__main__":
    unittest.main()
