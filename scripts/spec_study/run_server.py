#!/usr/bin/env python3
"""Run fresh llama-server instances for the speculation study."""

import argparse
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional


ROOT = Path(__file__).resolve().parents[2]
WORKLOAD = ROOT / "data" / "spec_workload.jsonl"
RAW_DIR = Path("/tmp/spec_study")
MODEL = Path("/Users/a470718/local-llm-bench/models/Qwen2.5-7B-Instruct-Q4_K_M.gguf")
SEED = 20260925
PORT = 18080
CONFIGS: Dict[str, List[str]] = {
    "none": ["--spec-type", "none"],
    "ngram_mod": ["--spec-type", "ngram-mod"],
    "ngram_simple": ["--spec-type", "ngram-simple"],
    "ngram_cache": ["--spec-type", "ngram-cache"],
}


def request(path: str, payload: Dict[str, Any], timeout: int = 180) -> Dict[str, Any]:
    """Send a JSON request to the local server."""
    body = json.dumps(payload).encode("utf-8")
    try:
        with urllib.request.urlopen(
            urllib.request.Request(
                f"http://127.0.0.1:{PORT}{path}",
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            ),
            timeout=timeout,
        ) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        raise RuntimeError(
            f"{path}: HTTP {error.code}: {error.read().decode('utf-8')}"
        ) from error


def wait_for_server(process: subprocess.Popen[Any], log: Any) -> None:
    """Wait until the health endpoint responds or surface early process failure."""
    deadline = time.monotonic() + 300
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"llama-server exited early; see {log.name}")
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{PORT}/health", timeout=2
            ) as response:
                if response.status == 200:
                    return
        except urllib.error.URLError:
            time.sleep(1)
    raise RuntimeError("llama-server did not become healthy in 300 seconds")


def tokenize_content(content: str) -> List[int]:
    """Tokenize text through the server's model tokenizer."""
    result = request("/tokenize", {"content": content})
    tokens = result.get("tokens", [])
    if not isinstance(tokens, list):
        raise RuntimeError(f"unexpected /tokenize result: {result}")
    return [int(token) for token in tokens]


def render_prompt(messages: List[Dict[str, str]]) -> str:
    """Render messages with the loaded model's template."""
    result = request(
        "/apply-template", {"messages": messages, "add_generation_prompt": True}
    )
    content = result.get("prompt", result.get("content"))
    if not isinstance(content, str):
        raise RuntimeError(f"unexpected /apply-template result: {result}")
    return content


def extract_content(response: Dict[str, Any]) -> str:
    """Extract standard OpenAI completion text."""
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError(f"response has no choices: {response}")
    message = choices[0].get("message", {})
    content = message.get("content", choices[0].get("text", ""))
    if not isinstance(content, str):
        raise RuntimeError(f"response content is not text: {response}")
    return content


def run_config(name: str, records: List[Dict[str, Any]]) -> None:
    """Run one spec configuration against all requests on a fresh server."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    log_path = RAW_DIR / f"server_{name}.log"
    command = [
        "llama-server",
        "-m",
        str(MODEL),
        "--port",
        str(PORT),
        "--parallel",
        "1",
        "-ngl",
        "99",
        "--ctx-size",
        "16384",
        "--seed",
        str(SEED),
        "--temp",
        "0",
        "--no-webui",
        "--perf",
    ] + CONFIGS[name]
    print("starting", name, flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        try:
            wait_for_server(process, log)
            output_path = RAW_DIR / f"server_{name}.jsonl"
            with output_path.open("w", encoding="utf-8") as handle:
                for index, record in enumerate(records, start=1):
                    started = time.monotonic_ns()
                    response = request(
                        "/v1/chat/completions",
                        {
                            "messages": record["messages"],
                            "temperature": 0,
                            "seed": SEED,
                            "max_tokens": record["max_tokens"],
                            "stream": False,
                        },
                    )
                    elapsed_ms = (time.monotonic_ns() - started) / 1_000_000
                    text = extract_content(response)
                    prompt = render_prompt(record["messages"])
                    item = {
                        "id": record["id"],
                        "family": record["family"],
                        "turn": record["turn"],
                        "prompt": prompt,
                        "prompt_tokens": tokenize_content(prompt),
                        "output": text,
                        "output_tokens": tokenize_content(text),
                        "response": response,
                        "client_elapsed_ms": elapsed_ms,
                    }
                    handle.write(
                        json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n"
                    )
                    handle.flush()
                    print(
                        f"{name} {index}/{len(records)} {record['id']} {elapsed_ms:.0f} ms",
                        flush=True,
                    )
        finally:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def main() -> None:
    """Run selected configs, regenerating the workload only when absent."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--configs", nargs="+", choices=list(CONFIGS), default=list(CONFIGS)
    )
    arguments = parser.parse_args()
    if not WORKLOAD.exists():
        raise SystemExit(
            f"workload missing: run {ROOT / 'scripts/spec_study/workload.py'} first"
        )
    if not MODEL.exists():
        raise SystemExit(f"model missing: {MODEL}")
    records = [
        json.loads(line)
        for line in WORKLOAD.read_text(encoding="utf-8").splitlines()
        if line
    ]
    for name in arguments.configs:
        run_config(name, records)


if __name__ == "__main__":
    main()
