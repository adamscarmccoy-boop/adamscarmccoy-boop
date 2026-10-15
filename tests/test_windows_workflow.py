"""Tests for scripts/windows_workflow.py."""

import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from scripts.windows_workflow import WindowsWorkflowOrchestrator


class TestWindowsWorkflow(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.model_path = os.path.join(self.temp_dir.name, "workflow_model.onnx")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_workflow_execution_with_llm_mock(self):
        orchestrator = WindowsWorkflowOrchestrator(
            onnx_model_path=self.model_path,
            llm_mode="lm_studio",
        )

        with patch.object(orchestrator.llm_runner, "generate", return_value="Analysis complete."):
            res = orchestrator.execute_pipeline(
                raw_input=[1.0, 2.0, 3.0, 4.0],
                weights=[2.0, 2.0, 2.0, 2.0],
                bias=[0.5, 0.5, 0.5, 0.5],
            )

        self.assertEqual(len(res["scaled_input"]), 4)
        self.assertAlmostEqual(res["dot_score"], 45.0)  # [1,2,3,4] . [1.5, 3.0, 4.5, 6.0] = 1.5 + 6.0 + 13.5 + 24.0 = 45.0
        self.assertEqual(len(res["onnx_output"]), 4)
        self.assertEqual(res["llm_response"], "Analysis complete.")

    def test_workflow_execution_llm_fallback(self):
        orchestrator = WindowsWorkflowOrchestrator(
            onnx_model_path=self.model_path,
            llm_mode="lm_studio",
        )

        with patch.object(orchestrator.llm_runner, "generate", side_effect=Exception("Connection refused")):
            res = orchestrator.execute_pipeline(
                raw_input=[1.0, 2.0, 3.0, 4.0],
                weights=[2.0, 2.0, 2.0, 2.0],
                bias=[0.5, 0.5, 0.5, 0.5],
            )

        self.assertIn("[LLM Offline / Fallback]", res["llm_response"])


if __name__ == "__main__":
    unittest.main()
