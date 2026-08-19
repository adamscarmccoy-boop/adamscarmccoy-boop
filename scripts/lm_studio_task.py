#!/usr/bin/env python3
"""Run a GitHub Actions task against an LM Studio OpenAI-compatible server."""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Send a prompt to LM Studio's OpenAI-compatible chat API."
    )
    parser.add_argument("--base-url", required=True, help="Base URL, for example http://127.0.0.1:1234/v1")
    parser.add_argument("--api-key", required=True, help="API key configured for LM Studio")
    parser.add_argument("--model", required=True, help="Model identifier loaded in LM Studio")
    parser.add_argument("--prompt", required=True, help="Prompt to send")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    endpoint = args.base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": args.model,
        "messages": [
            {
                "role": "system",
                "content": "You are running inside a GitHub Actions task. Be concise and actionable.",
            },
            {"role": "user", "content": args.prompt},
        ],
        "temperature": 0.2,
    }

    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {args.api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            response_body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        print(error.read().decode("utf-8"), file=sys.stderr)
        return 1
    except (urllib.error.URLError, TimeoutError) as error:
        print(f"Could not reach LM Studio at {endpoint}: {error}", file=sys.stderr)
        return 1

    try:
        content = response_body["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        print(json.dumps(response_body, indent=2), file=sys.stderr)
        return 1

    print(content)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
