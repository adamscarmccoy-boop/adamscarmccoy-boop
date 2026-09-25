"""Tests for scripts/lm_studio_task.py."""

import io
import json
import urllib.error
from unittest.mock import MagicMock, patch

import pytest

from scripts.lm_studio_task import main, parse_args


def test_parse_args():
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
        assert args.base_url == "http://localhost:1234/v1"
        assert args.api_key == "test-key"
        assert args.model == "test-model"
        assert args.prompt == "Hello world"


def test_main_success(capsys):
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

    with patch("sys.argv", test_args):
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.return_value.__enter__.return_value = mock_response
            exit_code = main()

    assert exit_code == 0
    captured = capsys.readouterr()
    assert "Response content from model" in captured.out


def test_main_http_error(capsys):
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

    with patch("sys.argv", test_args):
        with patch("urllib.request.urlopen", side_effect=http_error):
            exit_code = main()

    assert exit_code == 1
    captured = capsys.readouterr()
    assert "Bad Request Error Payload" in captured.err


def test_main_url_error(capsys):
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

    with patch("sys.argv", test_args):
        with patch("urllib.request.urlopen", side_effect=url_error):
            exit_code = main()

    assert exit_code == 1
    captured = capsys.readouterr()
    assert "Could not reach LM Studio at http://localhost:1234/v1/chat/completions: <urlopen error Connection refused>" in captured.err


def test_main_malformed_response_json(capsys):
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

    with patch("sys.argv", test_args):
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.return_value.__enter__.return_value = mock_response
            exit_code = main()

    assert exit_code == 1
    captured = capsys.readouterr()
    assert "Invalid payload format" in captured.err
