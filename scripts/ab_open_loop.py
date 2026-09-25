"""Open-loop A/B: requests arrive on a fixed schedule while others are generating.

Usage:
    python3 scripts/ab_open_loop.py --model model.gguf --corpus text.txt --rounds 3 \
        --target static=/path/to/old/radixforge --target continuous=build/radixforge \
        [--llama-server]

Each target is started fresh per round (order rotated), receives one warm-up request,
then N streaming requests at a fixed interval. Reports TTFT and end-to-end
percentiles; empty or "[ERROR" completions count as failures.
"""

import argparse
import json
import random
import statistics as st
import subprocess
import sys
import threading
import time
import urllib.request

PORT = 8460
N = 64
INTERVAL_S = 0.06

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--corpus", required=True)
ap.add_argument("--rounds", type=int, default=3)
ap.add_argument("--target", action="append", default=[], metavar="NAME=RADIXFORGE_BIN")
ap.add_argument(
    "--llama-server", action="store_true", help="also run llama-server --cache-reuse"
)
ARGS = ap.parse_args()
MODEL = ARGS.model
CORPUS = ARGS.corpus
TARGETS = {}
for spec in ARGS.target:
    name, binary = spec.split("=", 1)
    TARGETS[name] = [
        binary,
        "-m",
        MODEL,
        "--host",
        "127.0.0.1",
        "--port",
        str(PORT),
        "--ctx-size",
        "32768",
        "--max-seq",
        "32",
        "-ngl",
        "99",
    ]
if ARGS.llama_server:
    TARGETS["llama-server-cache"] = [
        "llama-server",
        "-m",
        MODEL,
        "--host",
        "127.0.0.1",
        "--port",
        str(PORT),
        "--ctx-size",
        "32768",
        "--parallel",
        "16",
        "--n-gpu-layers",
        "99",
        "--no-webui",
        "--cache-reuse",
        "256",
    ]

words = open(CORPUS).read().split()
SYSTEM = "You are an energy-market analyst. Reference material:\n" + " ".join(
    words[i % len(words)] for i in range(600)
)
rng = random.Random(20260925)
REQS = [
    (
        f"Question {i}: explain one factor behind electricity price changes, case {rng.randint(1, 999)}.",
        rng.choice([16, 32, 64, 96]),
    )
    for i in range(N)
]


def one(i, prompt, max_tokens, out):
    body = json.dumps(
        {
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": max_tokens,
            "temperature": 0,
            "stream": True,
        }
    ).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.monotonic()
    ttft = None
    text = ""
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            for raw in r:
                line = raw.decode().strip()
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                ev = json.loads(line[6:])
                delta = (ev.get("choices") or [{}])[0].get("delta", {}).get("content")
                if delta:
                    if ttft is None:
                        ttft = time.monotonic() - t0
                    text += delta
        ok = bool(text.strip()) and not text.strip().startswith("[ERROR")
    except Exception as e:  # noqa: BLE001
        ok = False
        text = str(e)
    out[i] = (ok, ttft, time.monotonic() - t0)


def run(name):
    log = open(f"/tmp/ab-open-loop-{name}.log", "w")
    p = subprocess.Popen(TARGETS[name], stdout=log, stderr=subprocess.STDOUT)
    for _ in range(600):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=1)
            break
        except Exception:  # noqa: BLE001
            time.sleep(0.2)
    one(-1, "Say OK.", 4, {})  # warmup
    out = {}
    threads = []
    start = time.monotonic()
    for i, (prompt, mt) in enumerate(REQS):
        delay = start + i * INTERVAL_S - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        t = threading.Thread(target=one, args=(i, prompt, mt, out))
        t.start()
        threads.append(t)
    for t in threads:
        t.join()
    wall = time.monotonic() - start
    p.terminate()
    p.wait(timeout=30)
    oks = [v for v in out.values() if v[0]]
    tt = sorted(v[1] for v in oks)
    e2e = sorted(v[2] for v in oks)
    q = lambda xs, f: xs[min(len(xs) - 1, int(f * len(xs)))] * 1000  # noqa: E731
    return {
        "target": name,
        "ok": len(oks),
        "n": N,
        "ttft_p50": q(tt, 0.5),
        "ttft_p95": q(tt, 0.95),
        "e2e_p50": q(e2e, 0.5),
        "e2e_p95": q(e2e, 0.95),
        "wall_s": wall,
    }


if __name__ == "__main__":
    names = list(TARGETS)
    if not names:
        sys.exit("no targets: pass --target NAME=BIN and/or --llama-server")
    results = []
    for rot in range(ARGS.rounds):
        k = rot % len(names)
        for name in names[k:] + names[:k]:
            r = run(name)
            r["round"] = rot
            results.append(r)
            print(json.dumps(r), flush=True)
            time.sleep(3)
    print("\nmedian over rounds:")
    for name in names:
        rs = [r for r in results if r["target"] == name]
        print(
            f"{name:24} ok {sum(r['ok'] for r in rs)}/{sum(r['n'] for r in rs)}  "
            + "  ".join(
                f"{k} {st.median(r[k] for r in rs):7.1f}"
                for k in ("ttft_p50", "ttft_p95", "e2e_p50", "e2e_p95", "wall_s")
            )
        )
