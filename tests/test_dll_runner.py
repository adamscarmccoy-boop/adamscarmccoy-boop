"""Tests for scripts/dll_runner.py."""

import unittest
from scripts.dll_runner import NativeBridgeRunner


class TestDllRunner(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runner = NativeBridgeRunner()

    def test_dot_product(self):
        vec_a = [1.0, 2.0, 3.0]
        vec_b = [4.0, 5.0, 6.0]
        res = self.runner.dot_product(vec_a, vec_b)
        self.assertAlmostEqual(res, 32.0)

    def test_vector_scale(self):
        vec = [1.0, 2.0, 3.0]
        res = self.runner.vector_scale(vec, 2.5)
        self.assertEqual(res, [2.5, 5.0, 7.5])

    def test_vector_add(self):
        vec_a = [1.0, 2.0, 3.0]
        vec_b = [0.5, 1.5, 2.5]
        res = self.runner.vector_add(vec_a, vec_b)
        self.assertEqual(res, [1.5, 3.5, 5.5])

    def test_compute_loss(self):
        actual = [1.0, 2.0, 3.0]
        predicted = [1.1, 1.9, 3.2]
        loss = self.runner.compute_loss(actual, predicted)
        # diffs: 0.1, -0.1, 0.2 -> sq: 0.01, 0.01, 0.04 -> sum: 0.06 -> mean: 0.02
        self.assertAlmostEqual(loss, 0.02)

    def test_mismatched_vector_lengths(self):
        with self.assertRaises(ValueError):
            self.runner.dot_product([1.0], [1.0, 2.0])


if __name__ == "__main__":
    unittest.main()
