#!/usr/bin/env python3
"""Exercise cancellation, payload limits, cache disabling, and shutdown."""

from __future__ import annotations

import argparse
import http.client
import json
import signal
import socket
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

REQUEST_TIMEOUT_SECONDS = 30
STARTUP_TIMEOUT_SECONDS = 90


def unused_local_port() -> int:
    """Return a currently unused loopback TCP port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def start_server(
    server: Path, model: Path, port: int, no_prefix_cache: bool = False
) -> subprocess.Popen[str]:
    """Start a CPU-only server configured for the robustness checks."""
    command = [
        str(server),
        "--model",
        str(model),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--ctx-size",
        "512",
        "--batch-size",
        "128",
        "--max-seq",
        "8",
        "--max-body-bytes",
        "1024",
        "--shutdown-timeout",
        "5",
        "--n-gpu-layers",
        "0",
        "--log-level",
        "warn",
    ]
    if no_prefix_cache:
        command.append("--no-prefix-cache")
    return subprocess.Popen(command, text=True)


def wait_for_server(port: int, process: subprocess.Popen[str]) -> None:
    """Wait for a healthy server or report an early exit."""
    deadline = time.monotonic() + STARTUP_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"server exited during startup with {process.returncode}"
            )
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/health", timeout=1
            ) as response:
                if response.status == 200 and json.load(response)["status"] == "ok":
                    return
        except urllib.error.URLError:
            time.sleep(0.2)
    raise TimeoutError("server did not become healthy")


def completion(port: int, prompt: str) -> str:
    """Request a small non-streaming completion."""
    payload = json.dumps(
        {"prompt": prompt, "max_tokens": 8, "temperature": 0, "stream": False}
    ).encode()
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
        return json.load(response)["choices"][0]["text"]


def metrics(port: int) -> dict[str, float]:
    """Read Prometheus samples into a numeric mapping."""
    with urllib.request.urlopen(
        f"http://127.0.0.1:{port}/metrics", timeout=5
    ) as response:
        lines = response.read().decode().splitlines()
    return {
        name: float(value)
        for line in lines
        if line and not line.startswith("#")
        for name, value in [line.split(maxsplit=1)]
    }


def assert_oversized_body_is_rejected(port: int) -> None:
    """Verify the configured payload maximum produces an OpenAI-style 413 response."""
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/completions",
        data=(b"x" * 2048),
        headers={"Content-Type": "application/json"},
    )
    try:
        urllib.request.urlopen(request, timeout=5)
    except urllib.error.HTTPError as error:
        if error.code != 413:
            raise AssertionError(
                f"oversized body returned {error.code}, expected 413"
            ) from error
        if "error" not in json.load(error):
            raise AssertionError(
                "413 response was not an OpenAI-style error envelope"
            ) from error
    else:
        raise AssertionError("oversized body was accepted")


def disconnect_stream(port: int) -> None:
    """Start streaming generation, read headers, then close the peer socket."""
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    body = json.dumps(
        {
            "prompt": "Once upon a time " * 24,
            "max_tokens": 128,
            "temperature": 0,
            "stream": True,
        }
    )
    connection.request(
        "POST", "/v1/completions", body, {"Content-Type": "application/json"}
    )
    response = connection.getresponse()
    if response.status != 200:
        raise AssertionError(f"stream setup returned {response.status}")
    response.read(1)
    connection.close()


def wait_for_cancellation(port: int, previous: float) -> None:
    """Wait until the server counts the intentionally disconnected request."""
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        if metrics(port).get("radixforge_cancelled_requests_total", 0) > previous:
            return
        time.sleep(0.1)
    raise AssertionError(
        "disconnected stream did not increment cancelled request metric"
    )


def stop_cleanly(process: subprocess.Popen[str]) -> None:
    """Request graceful termination and require a successful exit status."""
    process.send_signal(signal.SIGTERM)
    try:
        status = process.wait(timeout=15)
    except subprocess.TimeoutExpired as error:
        process.kill()
        process.wait()
        raise AssertionError("server did not exit within shutdown timeout") from error
    if status != 0:
        raise AssertionError(f"server exited with {status}, expected 0")


def main() -> None:
    """Run the robustness checks against a supplied server and tiny model."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    arguments = parser.parse_args()
    if not arguments.server.is_file() or not arguments.model.is_file():
        raise FileNotFoundError("server binary or model is missing")

    port = unused_local_port()
    process = start_server(arguments.server, arguments.model, port)
    try:
        wait_for_server(port, process)
        assert_oversized_body_is_rejected(port)
        before = metrics(port).get("radixforge_cancelled_requests_total", 0)
        disconnect_stream(port)
        wait_for_cancellation(port, before)
        if not completion(port, "A small lighthouse stood beside the sea.").strip():
            raise AssertionError("following completion was empty")
        stop_cleanly(process)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()

    no_cache_port = unused_local_port()
    no_cache_process = start_server(
        arguments.server, arguments.model, no_cache_port, True
    )
    try:
        wait_for_server(no_cache_port, no_cache_process)
        prompt = "A quiet village waited for dawn."
        completion(no_cache_port, prompt)
        completion(no_cache_port, prompt)
        if metrics(no_cache_port).get("radixforge_cache_hit_tokens_total", -1) != 0:
            raise AssertionError("--no-prefix-cache recorded cache-hit tokens")
        stop_cleanly(no_cache_process)
    finally:
        if no_cache_process.poll() is None:
            no_cache_process.kill()
            no_cache_process.wait()


if __name__ == "__main__":
    main()
