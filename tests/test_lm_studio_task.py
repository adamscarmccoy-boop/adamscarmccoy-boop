"""Tests for scripts/lm_studio_task.py."""

import io
import json
import sys
import unittest
import urllib.error
from unittest.mock import MagicMock, patch

from scripts.lm_studio_task import main, parse_args


class TestLmStudioTask(unittest.TestCase):
    def test_parse_args(self):
        test_args = [
            "lm_studio_task.py",
            "--base-url",
            "http://localhost:1234/v1",
            "--api-key",
            "test-key",
            "--model",
            "test-model",
            "--prompt",
            "Hello world",
        ]
        with patch("sys.argv", test_args):
            args = parse_args()
            self.assertEqual(args.base_url, "http://localhost:1234/v1")
            self.assertEqual(args.api_key, "test-key")
            self.assertEqual(args.model, "test-model")
            self.assertEqual(args.prompt, "Hello world")

    def test_main_success(self):
        test_args = [
            "lm_studio_task.py",
            "--base-url",
            "http://localhost:1234/v1/",
            "--api-key",
            "test-key",
            "--model",
            "test-model",
            "--prompt",
            "Hello world",
        ]
        response_payload = {
            "choices": [
                {
                    "message": {
                        "content": "Response content from model",
                    }
                }
            ]
        }
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps(response_payload).encode("utf-8")

        with patch("sys.argv", test_args), patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
            with patch("urllib.request.urlopen") as mock_urlopen:
                mock_urlopen.return_value.__enter__.return_value = mock_response
                exit_code = main()

        self.assertEqual(exit_code, 0)
        self.assertIn("Response content from model", mock_stdout.getvalue())

    def test_main_http_error(self):
        test_args = [
            "lm_studio_task.py",
            "--base-url",
            "http://localhost:1234/v1",
            "--api-key",
            "test-key",
            "--model",
            "test-model",
            "--prompt",
            "Hello world",
        ]
        http_error = urllib.error.HTTPError(
            url="http://localhost:1234/v1/chat/completions",
            code=400,
            msg="Bad Request",
            hdrs={},
            fp=io.BytesIO(b"Bad Request Error Payload"),
        )

        with patch("sys.argv", test_args), patch("sys.stderr", new_callable=io.StringIO) as mock_stderr:
            with patch("urllib.request.urlopen", side_effect=http_error):
                exit_code = main()

        self.assertEqual(exit_code, 1)
        self.assertIn("Bad Request Error Payload", mock_stderr.getvalue())

    def test_main_url_error(self):
        test_args = [
            "lm_studio_task.py",
            "--base-url",
            "http://localhost:1234/v1",
            "--api-key",
            "test-key",
            "--model",
            "test-model",
            "--prompt",
            "Hello world",
        ]
        url_error = urllib.error.URLError("Connection refused")

        with patch("sys.argv", test_args), patch("sys.stderr", new_callable=io.StringIO) as mock_stderr:
            with patch("urllib.request.urlopen", side_effect=url_error):
                exit_code = main()

        self.assertEqual(exit_code, 1)
        self.assertIn(
            "Could not reach LM Studio at http://localhost:1234/v1/chat/completions: <urlopen error Connection refused>",
            mock_stderr.getvalue(),
        )

    def test_main_malformed_response_json(self):
        test_args = [
            "lm_studio_task.py",
            "--base-url",
            "http://localhost:1234/v1",
            "--api-key",
            "test-key",
            "--model",
            "test-model",
            "--prompt",
            "Hello world",
        ]
        response_payload = {"error": "Invalid payload format"}
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps(response_payload).encode("utf-8")

        with patch("sys.argv", test_args), patch("sys.stderr", new_callable=io.StringIO) as mock_stderr:
            with patch("urllib.request.urlopen") as mock_urlopen:
                mock_urlopen.return_value.__enter__.return_value = mock_response
                exit_code = main()

        self.assertEqual(exit_code, 1)
        self.assertIn("Invalid payload format", mock_stderr.getvalue())

    def test_main_timeout_error(self):
        test_args = [
            "lm_studio_task.py",
            "--base-url",
            "http://localhost:1234/v1",
            "--api-key",
            "test-key",
            "--model",
            "test-model",
            "--prompt",
            "Hello world",
        ]
        timeout_error = TimeoutError("Request timed out")

        with patch("sys.argv", test_args), patch("sys.stderr", new_callable=io.StringIO) as mock_stderr:
            with patch("urllib.request.urlopen", side_effect=timeout_error):
                exit_code = main()

        self.assertEqual(exit_code, 1)
        self.assertIn(
            "Could not reach LM Studio at http://localhost:1234/v1/chat/completions: Request timed out",
            mock_stderr.getvalue(),
        )


if __name__ == "__main__":
    unittest.main()
