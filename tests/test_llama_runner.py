"""Tests for scripts/llama_runner.py."""

import io
import json
import unittest
import urllib.error
from unittest.mock import MagicMock, patch

from scripts.llama_runner import UnifiedLlamaRunner, query_openai_api, run_llama_cli


class TestLlamaRunner(unittest.TestCase):
    def test_query_openai_api_success(self):
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "choices": [{"message": {"content": "Hello from server"}}]
        }).encode("utf-8")

        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.return_value.__enter__.return_value = mock_response
            result = query_openai_api(
                base_url="http://127.0.0.1:1234/v1",
                prompt="Test prompt",
            )
            self.assertEqual(result, "Hello from server")

    def test_query_openai_api_error(self):
        http_error = urllib.error.HTTPError(
            url="http://127.0.0.1:1234/v1/chat/completions",
            code=500,
            msg="Internal Error",
            hdrs={},
            fp=io.BytesIO(b"Server failure"),
        )
        with patch("urllib.request.urlopen", side_effect=http_error):
            with self.assertRaises(RuntimeError):
                query_openai_api("http://127.0.0.1:1234/v1", "Test prompt")

    def test_run_llama_cli_success(self):
        mock_proc = MagicMock()
        mock_proc.stdout = "Response from llama.exe"

        with patch("subprocess.run", return_value=mock_proc) as mock_run:
            out = run_llama_cli("C:\\bin\\llama.exe", "C:\\models\\model.gguf", "Hello")
            self.assertEqual(out, "Response from llama.exe")
            mock_run.assert_called_once()

    def test_unified_runner_routing(self):
        runner_server = UnifiedLlamaRunner(mode="lm_studio")
        with patch("scripts.llama_runner.query_openai_api", return_value="Server output"):
            self.assertEqual(runner_server.generate("Prompt"), "Server output")

        runner_cli = UnifiedLlamaRunner(
            mode="llama_exe",
            llama_exe_path="/bin/llama.exe",
            gguf_model_path="/models/m.gguf",
        )
        with patch("scripts.llama_runner.run_llama_cli", return_value="CLI output"):
            self.assertEqual(runner_cli.generate("Prompt"), "CLI output")


if __name__ == "__main__":
    unittest.main()
