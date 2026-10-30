#!/usr/bin/env python3
"""Benchmark RadixForge cold and retained-prefix operation on a fixed workload."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

SYSTEM = " ".join(
    [
        "You are a concise research assistant. Use the following shared reference",
        "when answering. The reference describes a fictional power-grid planning",
        "exercise with generation, transmission, storage, reliability, demand,",
        "weather, market, policy, and maintenance constraints.",
    ]
    * 8
)
QUESTIONS = [
    f"Summarize the reliability risk for scenario {index} in one sentence."
    for index in range(24)
]


def percentile(values: list[float], fraction: float) -> float:
    """Return a nearest-rank percentile from a non-empty list."""
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(fraction * len(ordered)))]


def stream_one(
    port: int,
    system: str,
    question: str,
    output: dict[int, tuple[float, float]],
    index: int,
) -> None:
    """Run one streaming chat request and record first-token and completion latency."""
    body = {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": question},
        ],
        "max_tokens": 16,
        "temperature": 0,
        "stream": True,
    }
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    started = time.monotonic()
    first_token: float | None = None
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            for raw in response:
                line = raw.decode("utf-8").strip()
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                event = json.loads(line[6:])
                delta = (
                    (event.get("choices") or [{}])[0].get("delta", {}).get("content")
                )
                if delta and first_token is None:
                    first_token = time.monotonic() - started
        if first_token is None:
            raise RuntimeError("stream ended without content")
        output[index] = (first_token, time.monotonic() - started)
    except (
        OSError,
        RuntimeError,
        urllib.error.URLError,
        json.JSONDecodeError,
    ) as error:
        raise RuntimeError(f"request {index} failed: {error}") from error


def wait_for_server(port: int, process: subprocess.Popen[str]) -> None:
    """Wait for the health endpoint or surface an early process failure."""
    for _ in range(150):
        if process.poll() is not None:
            raise RuntimeError("server exited before accepting requests")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1):
                return
        except urllib.error.URLError:
            time.sleep(0.2)
    raise RuntimeError("server did not become healthy")


def run_phase(
    port: int, concurrency: int, shared_prefix: bool
) -> dict[str, float | int]:
    """Run the deterministic workload with shared or deliberately unique prefixes."""
    results: dict[int, tuple[float, float]] = {}
    started = time.monotonic()
    for offset in range(0, len(QUESTIONS), concurrency):
        threads = []
        for index, question in enumerate(
            QUESTIONS[offset : offset + concurrency], offset
        ):
            system = SYSTEM if shared_prefix else f"Request identity {index}. {SYSTEM}"
            threads.append(
                threading.Thread(
                    target=stream_one,
                    args=(port, system, question, results, index),
                )
            )
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    wall_seconds = time.monotonic() - started
    ttft = [value[0] * 1000 for value in results.values()]
    e2e = [value[1] * 1000 for value in results.values()]
    if len(results) != len(QUESTIONS):
        raise RuntimeError(f"completed {len(results)} of {len(QUESTIONS)} requests")
    return {
        "requests": len(results),
        "ttft_p50_ms": percentile(ttft, 0.5),
        "ttft_p95_ms": percentile(ttft, 0.95),
        "e2e_p50_ms": percentile(e2e, 0.5),
        "e2e_p95_ms": percentile(e2e, 0.95),
        "throughput_req_s": len(results) / wall_seconds,
        "wall_s": wall_seconds,
    }


def warmup(port: int) -> None:
    """Populate the shared system-prompt prefix without including it in results."""
    output: dict[int, tuple[float, float]] = {}
    stream_one(port, SYSTEM, "Acknowledge the reference in one word.", output, 0)


def metrics(port: int) -> dict[str, float]:
    """Read the numeric Prometheus samples exposed by the server."""
    with urllib.request.urlopen(
        f"http://127.0.0.1:{port}/metrics", timeout=10
    ) as response:
        text = response.read().decode("utf-8")
    values: dict[str, float] = {}
    for line in text.splitlines():
        if line and not line.startswith("#"):
            name, value = line.split(maxsplit=1)
            values[name] = float(value)
    return values


def start_server(args: argparse.Namespace, log_path: Path) -> subprocess.Popen[str]:
    """Start a fresh RadixForge process for one cold or warm phase."""
    command = [
        args.binary,
        "-m",
        args.model,
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
        "--ctx-size",
        "8192",
        "--batch-size",
        "512",
        "--max-seq",
        "32",
        "--admit-window-ms",
        "0",
        "--request-timeout",
        "120",
        "--kv-high-watermark",
        "0.9",
        "-ngl",
        "99",
    ]
    return subprocess.Popen(
        command, stdout=log_path.open("w"), stderr=subprocess.STDOUT, text=True
    )


def main() -> int:
    """Run fresh-server cold and retained-prefix phases and write JSON results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", default="build/radixforge")
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--port", type=int, default=18500)
    args = parser.parse_args()

    report: dict[str, Any] = {
        "workload": {"requests": len(QUESTIONS), "max_tokens": 16}
    }
    for concurrency in (1, 4):
        phases: dict[str, dict[str, float | int]] = {}
        for phase, shared_prefix in (("low_reuse", False), ("warm", True)):
            log = args.out.with_name(f"benchmark-{phase}-c{concurrency}.log")
            process = start_server(args, log)
            try:
                wait_for_server(args.port, process)
                if shared_prefix:
                    warmup(args.port)
                before = metrics(args.port)
                result = run_phase(args.port, concurrency, shared_prefix)
                after = metrics(args.port)
                prompt_delta = (
                    after["radixforge_prompt_tokens_total"]
                    - before["radixforge_prompt_tokens_total"]
                )
                hit_delta = (
                    after["radixforge_cache_hit_tokens_total"]
                    - before["radixforge_cache_hit_tokens_total"]
                )
                result["cache_hit_token_ratio"] = (
                    hit_delta / prompt_delta if prompt_delta else 0.0
                )
                result["cache_hit_tokens"] = hit_delta
                result["prompt_tokens"] = prompt_delta
                phases[phase] = result
            finally:
                process.terminate()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
        report[f"concurrency_{concurrency}"] = phases
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
