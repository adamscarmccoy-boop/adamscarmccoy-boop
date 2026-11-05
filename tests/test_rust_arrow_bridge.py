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

try:
    import duckdb
except ImportError:
    duckdb = None

try:
    import lancedb
except ImportError:
    lancedb = None

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

    def test_score_table_vectors_aot(self):
        table = pa.Table.from_pydict({
            "id": [1, 2, 3],
            "vector": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.5, 0.5, 0.0]]
        })
        query = [1.0, 0.0, 0.0]
        scored_table = self.runner.score_table_vectors_aot(table, query_vector=query)
        self.assertIn("rust_aot_score", scored_table.column_names)
        scores = scored_table.column("rust_aot_score").to_pylist()
        self.assertAlmostEqual(scores[0], 1.0)
        self.assertAlmostEqual(scores[1], 0.0)
        self.assertAlmostEqual(scores[2], 0.5)

    def test_duckdb_loading_and_query(self):
        if duckdb is None:
            self.skipTest("DuckDB not installed")
        table = pa.Table.from_pydict({
            "id": [1, 2, 3],
            "metric": [10.0, 20.0, 30.0]
        })
        con = self.runner.load_to_duckdb(table, table_name="test_arrow")
        res = con.sql("SELECT SUM(metric) as total FROM test_arrow").arrow().read_all()
        self.assertEqual(res.to_pydict()["total"][0], 60.0)

    def test_lancedb_vector_loading_and_search(self):
        if lancedb is None:
            self.skipTest("LanceDB not installed")
        with tempfile.TemporaryDirectory() as tmp_dir:
            table = pa.Table.from_pydict({
                "id": [1, 2],
                "vector": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
            })
            tbl = self.runner.load_to_lancedb(table, db_path=tmp_dir, table_name="test_lance")
            search_arrow = tbl.search([1.0, 2.0, 3.0]).limit(1).to_arrow()
            self.assertEqual(search_arrow.column("id").to_pylist(), [1])

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


class TestMmap808DuckDBLanceDB(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.mmap_path = os.path.join(self.temp_dir.name, "vectors_808.bin")
        self.runner = RustArrowBridgeRunner()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_mmap_808_bytes_creation_and_reading(self):
        # 101 f64 * 8 bytes = exactly 808 bytes
        created_path = self.runner.create_mmap_808_file(self.mmap_path, num_records=4, dim_f64=101)
        self.assertTrue(os.path.exists(created_path))
        self.assertEqual(os.path.getsize(created_path), 4 * 808)

        table, mm = self.runner.read_mmap_808_table(created_path, num_records=4, dim_f64=101)
        self.assertEqual(table.num_rows, 4)
        self.assertEqual(len(table.column("raw_808_bytes")[0].as_py()), 808)
        self.assertEqual(len(table.column("vector")[0].as_py()), 101)
        mm.close()

    def test_mmap_808_duckdb_query(self):
        if duckdb is None:
            self.skipTest("DuckDB not installed")
        self.runner.create_mmap_808_file(self.mmap_path, num_records=3, dim_f64=101)
        table, mm = self.runner.read_mmap_808_table(self.mmap_path, num_records=3, dim_f64=101)

        con = self.runner.load_to_duckdb(table, table_name="mmap_808_table")
        res = con.sql("SELECT COUNT(*) as cnt, MAX(id) as max_id FROM mmap_808_table").arrow().read_all()
        self.assertEqual(res.to_pydict()["cnt"][0], 3)
        self.assertEqual(res.to_pydict()["max_id"][0], 3)
        mm.close()

    def test_mmap_808_lancedb_and_rust_aot_scoring(self):
        if lancedb is None:
            self.skipTest("LanceDB not installed")
        self.runner.create_mmap_808_file(self.mmap_path, num_records=3, dim_f64=101)
        table, mm = self.runner.read_mmap_808_table(self.mmap_path, num_records=3, dim_f64=101)

        # LanceDB storage
        lance_dir = os.path.join(self.temp_dir.name, "lance_db")
        tbl = self.runner.load_to_lancedb(table, db_path=lance_dir, table_name="mmap_808_vectors")
        self.assertEqual(tbl.count_rows(), 3)

        # Rust AOT in-table model scoring on 101-element (808-byte) vector
        query_vec = [1.0] * 101
        scored_tbl = self.runner.score_table_vectors_aot(table, query_vector=query_vec)
        self.assertIn("rust_aot_score", scored_tbl.column_names)
        self.assertEqual(len(scored_tbl.column("rust_aot_score")), 3)
        mm.close()
