#!/usr/bin/env python3
"""Exercise deterministic generation across cache reuse and eviction."""

from __future__ import annotations

import argparse
import json
import socket
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

REQUEST_TIMEOUT_SECONDS = 90
STARTUP_TIMEOUT_SECONDS = 90
MAX_TOKENS = 24


def unused_local_port() -> int:
    """Return a currently unused loopback TCP port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def request_completion(port: int, prompt: str) -> str:
    """Request a greedy non-streaming completion and return its text."""
    payload = json.dumps(
        {
            "prompt": prompt,
            "max_tokens": MAX_TOKENS,
            "temperature": 0,
            "top_p": 1,
            "stream": False,
        }
    ).encode()
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
        data = json.load(response)
    return data["choices"][0]["text"]


def server_stats(port: int) -> dict[str, object]:
    """Return the server's cache metrics."""
    with urllib.request.urlopen(
        f"http://127.0.0.1:{port}/admin/stats", timeout=REQUEST_TIMEOUT_SECONDS
    ) as response:
        return json.load(response)


def wait_for_server(port: int, process: subprocess.Popen[str]) -> None:
    """Wait for the server health endpoint or surface its startup failure."""
    health_url = f"http://127.0.0.1:{port}/health"
    deadline = time.monotonic() + STARTUP_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"server exited during startup with {process.returncode}"
            )
        try:
            with urllib.request.urlopen(health_url, timeout=1):
                return
        except urllib.error.URLError:
            time.sleep(0.2)
    raise TimeoutError("server did not become healthy")


def assert_equal(expected: str, actual: str, label: str) -> None:
    """Raise with a compact, actionable completion diff when values differ."""
    if actual != expected:
        raise AssertionError(
            f"{label} differed from cold generation:\n"
            f"expected ({len(expected)} chars): {expected!r}\n"
            f"actual   ({len(actual)} chars): {actual!r}"
        )


def main() -> None:
    """Start the server and verify cache operations preserve greedy output."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    arguments = parser.parse_args()

    if not arguments.server.is_file():
        raise FileNotFoundError(f"server binary not found: {arguments.server}")
    if not arguments.model.is_file():
        raise FileNotFoundError(f"model not found: {arguments.model}")

    port = unused_local_port()
    command = [
        str(arguments.server),
        "--model",
        str(arguments.model),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--ctx-size",
        "512",
        "--batch-size",
        "128",
        "--max-seq",
        "5",
        "--n-gpu-layers",
        "0",
        "--log-level",
        "warn",
    ]
    prefix = "Once upon a time in a quiet forest, a small fox named Ember " * 8
    prompt = prefix + "finished this sentence with a surprising discovery:"
    siblings = [
        prefix + "met a curious rabbit and said:",
        prefix + "found an old map that showed:",
    ]
    eviction_prompts = [
        "A distant lighthouse keeper wrote in a journal:",
        "On a mountain trail, a patient guide explained:",
    ]

    process = subprocess.Popen(command, text=True)
    try:
        wait_for_server(port, process)
        cold = request_completion(port, prompt)
        if not cold or cold.startswith("[ERROR"):
            raise AssertionError(f"cold generation failed: {cold!r}")

        for sibling in siblings:
            request_completion(port, sibling)
        assert_equal(cold, request_completion(port, prompt), "cached generation")

        # Four usable sequence IDs make the second idle filler force LRU eviction.
        for eviction_prompt in eviction_prompts:
            request_completion(port, eviction_prompt)
        evictions = server_stats(port)["cache"]["evictions"]  # type: ignore[index]
        if evictions < 1:
            raise AssertionError(f"expected cache eviction, got {evictions}")
        assert_equal(cold, request_completion(port, prompt), "post-eviction generation")
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


if __name__ == "__main__":
    main()
