import json
import os
import shutil
import tempfile
import unittest
from unittest.mock import patch, MagicMock

from scripts.lm_studio_embed import (
    prepare_texts_from_jsonl,
    send_embedding_request,
)


class TestLMStudioEmbed(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_prepare_texts_from_jsonl(self):
        jsonl_file = os.path.join(self.test_dir, "dataset.jsonl")
        records = [
            {"text": "Sample text line 1", "metadata": {"file": "a.txt"}},
            {"messages": [{"role": "user", "content": "hello"}]}
        ]
        with open(jsonl_file, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        docs = prepare_texts_from_jsonl(jsonl_file)
        self.assertEqual(len(docs), 2)
        self.assertEqual(docs[0]["text"], "Sample text line 1")
        self.assertIn("USER: hello", docs[1]["text"])

    @patch("urllib.request.urlopen")
    def test_send_embedding_request(self, mock_urlopen):
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "data": [
                {"embedding": [0.1, 0.2, 0.3]},
                {"embedding": [0.4, 0.5, 0.6]}
            ]
        }).encode("utf-8")
        mock_response.__enter__.return_value = mock_response
        mock_urlopen.return_value = mock_response

        embeddings = send_embedding_request(
            base_url="http://127.0.0.1:1234/v1",
            api_key="lm-studio",
            model="text-embedding-llama-nemotron-embed-1b-v2",
            input_texts=["text 1", "text 2"]
        )

        self.assertEqual(len(embeddings), 2)
        self.assertEqual(embeddings[0], [0.1, 0.2, 0.3])


if __name__ == "__main__":
    unittest.main()
