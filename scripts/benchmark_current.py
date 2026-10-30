#!/usr/bin/env python3
"""Compare RadixForge and vendored llama-server on a shared-prefix workload."""

from __future__ import annotations

import argparse
import json
import subprocess
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

SYSTEM = " ".join(
    [
        "Reference: a fictional grid has generation, storage, reliability, demand,",
        "weather, market, policy, and maintenance constraints. Answer concisely.",
    ]
    * 8
)
QUESTIONS = [f" Scenario {index}: state one reliability risk." for index in range(24)]


def percentile(values: list[float], fraction: float) -> float:
    """Return a nearest-rank percentile."""
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(fraction * len(ordered)))]


def prompt(question: str) -> str:
    """Return the same raw prompt for every server implementation."""
    return f"{SYSTEM}\nQuestion:{question}\nAnswer:"


def request_stream(
    kind: str, port: int, text: str, cache_enabled: bool
) -> tuple[float, float, int, int]:
    """Issue one streaming request and return client timings plus cache accounting."""
    if kind == "radixforge":
        url = f"http://127.0.0.1:{port}/v1/completions"
        body: dict[str, Any] = {
            "prompt": text,
            "max_tokens": 16,
            "temperature": 0,
            "stream": True,
        }
    else:
        url = f"http://127.0.0.1:{port}/completion"
        body = {
            "prompt": text,
            "n_predict": 16,
            "temperature": 0,
            "stream": True,
            "cache_prompt": cache_enabled,
        }
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    started = time.monotonic()
    first: float | None = None
    cache_n = prompt_n = 0
    with urllib.request.urlopen(request, timeout=120) as response:
        for raw in response:
            line = raw.decode("utf-8").strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            event = json.loads(line[6:])
            content = (
                (event.get("choices") or [{}])[0].get("delta", {}).get("content")
                if kind == "radixforge"
                else event.get("content")
            )
            if content and first is None:
                first = time.monotonic() - started
            if kind == "llama" and event.get("timings"):
                cache_n = int(event["timings"]["cache_n"])
                prompt_n = int(event["timings"]["prompt_n"])
    if first is None:
        raise RuntimeError("stream ended without content")
    return first, time.monotonic() - started, cache_n, prompt_n


def wait_for_server(port: int, process: subprocess.Popen[str]) -> None:
    """Wait for the health endpoint or an early process failure."""
    for _ in range(180):
        if process.poll() is not None:
            raise RuntimeError("server exited before accepting requests")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1):
                return
        except urllib.error.URLError:
            time.sleep(0.2)
    raise RuntimeError("server did not become healthy")


def radix_metrics(port: int) -> dict[str, float]:
    """Read RadixForge Prometheus samples."""
    with urllib.request.urlopen(
        f"http://127.0.0.1:{port}/metrics", timeout=10
    ) as response:
        lines = response.read().decode().splitlines()
    return {
        name: float(value)
        for line in lines
        if line and not line.startswith("#")
        for name, value in [line.split(maxsplit=1)]
    }


def start(
    args: argparse.Namespace, kind: str, cache_enabled: bool, log: Path
) -> subprocess.Popen[str]:
    """Start one comparison server with matched context, slots, and Metal settings."""
    common = [
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
        "--ctx-size",
        "8192",
        "-ngl",
        "99",
    ]
    if kind == "radixforge":
        command = [
            args.radixforge,
            "-m",
            args.model,
            *common,
            "--batch-size",
            "512",
            "--max-seq",
            "32",
            "--admit-window-ms",
            "0",
            "--request-timeout",
            "120",
        ]
        if not cache_enabled:
            command.append("--no-prefix-cache")
    else:
        command = [
            args.llama_server,
            "-m",
            args.model,
            *common,
            "--parallel",
            "32",
            "--flash-attn",
            "auto",
            "--no-webui",
        ]
    return subprocess.Popen(
        command, stdout=log.open("w"), stderr=subprocess.STDOUT, text=True
    )


def phase(
    args: argparse.Namespace,
    kind: str,
    cache_enabled: bool,
    concurrency: int,
    repetition: int,
) -> dict[str, float | int]:
    """Measure one fresh-server phase."""
    label = f"{kind}-{'on' if cache_enabled else 'off'}-c{concurrency}-r{repetition}"
    process = start(args, kind, cache_enabled, args.out.parent / f"{label}.log")
    try:
        wait_for_server(args.port, process)
        request_stream(kind, args.port, prompt("Warmup."), cache_enabled)
        before = radix_metrics(args.port) if kind == "radixforge" else {}
        values: list[tuple[float, float, int, int]] = []
        started = time.monotonic()
        for offset in range(0, len(QUESTIONS), concurrency):
            batch: list[tuple[float, float, int, int] | None] = [None] * min(
                concurrency, len(QUESTIONS) - offset
            )

            def run_one(
                index: int,
                question: str,
                destination: list[tuple[float, float, int, int] | None] = batch,
            ) -> None:
                destination[index] = request_stream(
                    kind, args.port, prompt(question), cache_enabled
                )

            threads = [
                threading.Thread(target=run_one, args=(i, question))
                for i, question in enumerate(QUESTIONS[offset : offset + concurrency])
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            values.extend(value for value in batch if value is not None)
        wall = time.monotonic() - started
        after = radix_metrics(args.port) if kind == "radixforge" else {}
        ttft = [value[0] * 1000 for value in values]
        e2e = [value[1] * 1000 for value in values]
        if kind == "radixforge":
            cache_hits = (
                after["radixforge_cache_hit_tokens_total"]
                - before["radixforge_cache_hit_tokens_total"]
            )
            prompt_tokens = (
                after["radixforge_prompt_tokens_total"]
                - before["radixforge_prompt_tokens_total"]
            )
        else:
            cache_hits = sum(value[2] for value in values)
            prompt_tokens = cache_hits + sum(value[3] for value in values)
        return {
            "ttft_p50_ms": percentile(ttft, 0.5),
            "ttft_p95_ms": percentile(ttft, 0.95),
            "e2e_p50_ms": percentile(e2e, 0.5),
            "e2e_p95_ms": percentile(e2e, 0.95),
            "throughput_req_s": len(values) / wall,
            "cache_hit_ratio": cache_hits / prompt_tokens if prompt_tokens else 0.0,
        }
    finally:
        process.terminate()
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def main() -> None:
    """Run three repetitions of four conditions at two concurrency levels."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--radixforge", default="build/radixforge")
    parser.add_argument("--llama-server", default="build-llama-server/bin/llama-server")
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--port", type=int, default=18500)
    parser.add_argument("--repetitions", type=int, default=3)
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {}
    for concurrency in (1, 4):
        for kind, enabled in (
            ("radixforge", False),
            ("radixforge", True),
            ("llama", False),
            ("llama", True),
        ):
            name = f"{kind}_{'cache_on' if enabled else 'cache_off'}_c{concurrency}"
            report[name] = [
                phase(args, kind, enabled, concurrency, repetition)
                for repetition in range(args.repetitions)
            ]
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
