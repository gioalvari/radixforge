#!/usr/bin/env python3
"""Measure target verification cost by speculative draft length."""

import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List


MODEL = Path("/Users/a470718/local-llm-bench/models/Qwen2.5-7B-Instruct-Q4_K_M.gguf")
OUT = Path("/tmp/spec_study/cost_curve.json")


def first_json(text: str) -> Any:
    """Decode the first JSON document contained in llama-bench output."""
    start = min(
        (position for position in (text.find("{"), text.find("[")) if position >= 0),
        default=-1,
    )
    if start < 0:
        raise ValueError("no JSON document in llama-bench output")
    return json.JSONDecoder().raw_decode(text[start:])[0]


def numeric_rate(value: Any) -> float:
    """Find a token-rate field in a nested llama-bench JSON result."""
    if isinstance(value, dict):
        for key in ("avg_ts", "ts", "tokens_per_second"):
            if isinstance(value.get(key), (int, float)):
                return float(value[key])
        for child in value.values():
            try:
                return numeric_rate(child)
            except ValueError:
                pass
    if isinstance(value, list):
        for child in value:
            try:
                return numeric_rate(child)
            except ValueError:
                pass
    raise ValueError("no token-rate field in llama-bench JSON")


def main() -> None:
    """Run 1+k-token forward benchmarks and store absolute and relative times."""
    draft_lengths = [0, 1, 2, 4, 8, 16]
    command = [
        "llama-bench",
        "-m",
        str(MODEL),
        "-p",
        "1,2,3,5,9,17",
        "-n",
        "0",
        "-d",
        "1536",
        "-ngl",
        "99",
        "-r",
        "3",
        "-o",
        "json",
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    parsed = first_json(completed.stdout)
    # Prompt-only benchmarks report prompt token/s; test positions map to 1+k tokens.
    rows: List[Dict[str, float]] = []
    candidates: List[Any] = []
    if isinstance(parsed, list):
        candidates = parsed
    elif isinstance(parsed, dict):
        candidates = parsed.get("results", parsed.get("benchmarks", []))
    if not isinstance(candidates, list) or len(candidates) < len(draft_lengths):
        raise RuntimeError(f"unexpected llama-bench rows: {parsed}")
    for draft_length, row in zip(draft_lengths, candidates):
        rate = numeric_rate(row)
        tokens = draft_length + 1
        rows.append(
            {
                "draft_length": draft_length,
                "tokens": tokens,
                "tokens_per_second": rate,
                "time_ms": 1000 * tokens / rate,
            }
        )
    base = rows[0]["time_ms"]
    for row in rows:
        row["relative_cost"] = row["time_ms"] / base
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps({"command": command, "rows": rows, "raw": parsed}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
