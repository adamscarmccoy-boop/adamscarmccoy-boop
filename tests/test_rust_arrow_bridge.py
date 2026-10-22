"""Tests for scripts/rust_arrow_bridge.py."""

import os
import tempfile
import unittest
import numpy as np
import pyarrow as pa

try:
    import polars as pl
except ImportError:
    pl = None

from scripts.rust_arrow_bridge import RustArrowBridgeRunner
from scripts.windows_workflow import WindowsWorkflowOrchestrator


class TestRustArrowBridge(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runner = RustArrowBridgeRunner()

    def test_dot_product_pyarrow(self):
        vec_a = pa.array([1.0, 2.0, 3.0])
        vec_b = pa.array([4.0, 5.0, 6.0])
        res = self.runner.dot_product(vec_a, vec_b)
        self.assertAlmostEqual(res, 32.0)

    def test_vector_scale_pyarrow(self):
        arr = pa.array([1.0, 2.0, 3.0, 4.0])
        res = self.runner.vector_scale(arr, 2.5)
        self.assertEqual(res, [2.5, 5.0, 7.5, 10.0])

    def test_vector_add_pyarrow(self):
        vec_a = pa.array([1.0, 2.0, 3.0])
        vec_b = pa.array([0.5, 1.5, 2.5])
        res = self.runner.vector_add(vec_a, vec_b)
        self.assertEqual(res, [1.5, 3.5, 5.5])

    def test_compute_loss(self):
        actual = pa.array([1.0, 2.0, 3.0])
        predicted = pa.array([1.1, 1.9, 3.2])
        loss = self.runner.compute_loss(actual, predicted)
        self.assertAlmostEqual(loss, 0.02)

    def test_polars_series_input(self):
        if pl is None:
            self.skipTest("Polars not installed")
        s_a = pl.Series("a", [1.0, 2.0, 3.0])
        s_b = pl.Series("b", [2.0, 2.0, 2.0])
        res = self.runner.dot_product(s_a, s_b)
        self.assertAlmostEqual(res, 12.0)

    def test_prepare_onnx_tensor(self):
        arr = pa.array([1.0, 2.0, 3.0])
        tensor = self.runner.prepare_onnx_tensor(arr)
        self.assertEqual(tensor.shape, (1, 3))
        self.assertEqual(tensor.dtype, np.float32)
        np.testing.assert_allclose(tensor, np.array([[1.0, 2.0, 3.0]], dtype=np.float32))

    def test_process_arrow_table(self):
        table = pa.Table.from_arrays(
            [pa.array([1.0, 2.0, 3.0]), pa.array([10.0, 20.0, 30.0])],
            names=["col_a", "col_b"],
        )
        updated_table = self.runner.process_arrow_table(table, column_name="col_a", scale=3.0)
        self.assertIn("scaled_features", updated_table.column_names)
        self.assertEqual(updated_table.column("scaled_features").to_pylist(), [3.0, 6.0, 9.0])

    def test_mismatched_vector_lengths(self):
        vec_a = pa.array([1.0, 2.0])
        vec_b = pa.array([1.0, 2.0, 3.0])
        with self.assertRaises(ValueError):
            self.runner.dot_product(vec_a, vec_b)


class TestRustArrowWorkflowIntegration(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.model_path = os.path.join(self.temp_dir.name, "rust_arrow_workflow.onnx")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_orchestrator_with_rust_arrow(self):
        orchestrator = WindowsWorkflowOrchestrator(
            onnx_model_path=self.model_path,
            use_rust_arrow=True,
        )

        arrow_input = pa.array([1.0, 2.0, 3.0, 4.0])
        res = orchestrator.execute_pipeline(
            raw_input=arrow_input,
            weights=[2.0, 2.0, 2.0, 2.0],
            bias=[0.5, 0.5, 0.5, 0.5],
        )

        self.assertEqual(len(res["scaled_input"]), 4)
        self.assertAlmostEqual(res["dot_score"], 45.0)
        self.assertEqual(len(res["onnx_output"]), 4)
        self.assertIn("Loss:", res["metrics_summary"])


if __name__ == "__main__":
    unittest.main()
