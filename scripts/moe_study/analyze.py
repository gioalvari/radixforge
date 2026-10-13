# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=2"]
# ///
"""Prefix-conditioned expert residency: offline analysis of moe_trace output.

Question: does the radix-tree node a request matches predict which experts its
decode tokens will route to, well enough to beat LRU expert caching?

Why the node matters: with prefix caching the shared prefix is never prefilled
again, so its routing signal is lost unless the node stored it. Profiles:

- ``global``: decode routing of all earlier requests.
- ``node``: decode routing of earlier requests sharing a >= ``--min-lcp`` token
  prefix with this one (what a radix node can store).
- ``delta``: routing of this request's own uncached prompt tokens.
- ``node+delta``: sum of the two normalised histograms.
- ``full_prefill`` (reference): routing of the whole prompt, only available
  without prefix caching.

Outputs recall@C (fraction of decode routings covered by the profile's top-C
experts per layer) and a per-layer cache simulation over decode tokens with C
slots per layer: LRU, profile-prefetch + LRU, and Belady (oracle).
"""

from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Req:
    """One traced request."""

    id: str
    family: str
    agent: str
    tokens: np.ndarray  # prompt + generated
    n_prompt: int
    routes: np.ndarray  # [n_routed, L, k] uint8

    @property
    def prefill(self) -> np.ndarray:
        """Routing of prompt tokens."""
        return self.routes[: self.n_prompt]

    @property
    def decode(self) -> np.ndarray:
        """Routing of generated tokens (fed back through the model)."""
        return self.routes[self.n_prompt :]


def load(prefix: Path) -> tuple[dict, list[Req]]:
    """Load meta, rows and routing tensor written by ``moe_trace``."""
    meta = json.loads(prefix.with_suffix(".meta.json").read_text())
    n_layers, k = len(meta["moe_layers"]), meta["k"]
    raw = np.fromfile(prefix.with_suffix(".bin"), dtype=np.uint8)
    reqs = []
    for line in prefix.with_suffix(".jsonl").read_text().splitlines():
        row = json.loads(line)
        start, n = row["offset"], row["n_routed"] * n_layers * k
        routes = raw[start : start + n].reshape(row["n_routed"], n_layers, k)
        reqs.append(
            Req(
                id=row["id"],
                family=row["family"],
                agent=row["agent"],
                tokens=np.asarray(row["tokens"], dtype=np.int64),
                n_prompt=row["n_prompt"],
                routes=routes,
            )
        )
    return meta, reqs


def histogram(routes: np.ndarray, n_expert: int) -> np.ndarray:
    """Per-layer expert counts, shape [L, n_expert]."""
    n_layers = routes.shape[1]
    out = np.zeros((n_layers, n_expert), dtype=np.float64)
    if routes.size == 0:
        return out
    flat = routes.transpose(1, 0, 2).reshape(n_layers, -1)
    for layer in range(n_layers):
        out[layer] = np.bincount(flat[layer], minlength=n_expert)
    return out


def normalise(h: np.ndarray) -> np.ndarray:
    """Row-normalise a histogram, leaving empty rows at zero."""
    s = h.sum(axis=1, keepdims=True)
    return np.divide(h, s, out=np.zeros_like(h), where=s > 0)


def lcp(a: np.ndarray, b: np.ndarray) -> int:
    """Length of the longest common token prefix."""
    n = min(len(a), len(b))
    diff = np.nonzero(a[:n] != b[:n])[0]
    return int(diff[0]) if diff.size else n


def build_profiles(reqs: list[Req], n_expert: int, min_lcp: int) -> list[dict]:
    """Causal profiles for each request (only earlier requests are visible)."""
    n_layers = reqs[0].routes.shape[1]
    global_h = np.zeros((n_layers, n_expert))
    decode_h = [histogram(r.decode, n_expert) for r in reqs]
    out = []
    for i, r in enumerate(reqs):
        matches = [
            (lcp(r.tokens[: r.n_prompt], p.tokens[: p.n_prompt]), j)
            for j, p in enumerate(reqs[:i])
        ]
        best = max((m for m, _ in matches), default=0)
        cached = best if best >= min_lcp else 0
        node_h = np.zeros((n_layers, n_expert))
        if cached:
            for m, j in matches:
                if m >= cached:  # requests below the same radix node
                    node_h += decode_h[j]
        delta_h = histogram(r.prefill[cached:], n_expert)
        out.append(
            {
                "cached": cached,
                "global": normalise(global_h.copy()),
                "node": normalise(node_h),
                "delta": normalise(delta_h),
                "node+delta": normalise(node_h) + normalise(delta_h),
                "full_prefill": normalise(histogram(r.prefill, n_expert)),
            }
        )
        global_h += decode_h[i]
    return out


def recall_at(profile: np.ndarray, decode: np.ndarray, c: int) -> float:
    """Fraction of decode (token, layer, slot) routings inside the top-c set."""
    if decode.size == 0 or profile.sum() == 0:
        return float("nan")
    top = np.argsort(-profile, axis=1, kind="stable")[:, :c]
    member = np.zeros_like(profile, dtype=bool)
    np.put_along_axis(member, top, True, axis=1)
    layers = np.arange(decode.shape[1])[None, :, None]
    return float(member[layers, decode].mean())


def simulate(
    reqs: list[Req], profiles: list[dict], c: int, policy: str
) -> tuple[int, int, int]:
    """Per-layer cache over decode tokens. Returns (demand misses, prefetch loads, accesses)."""
    n_layers = reqs[0].routes.shape[1]
    caches = [OrderedDict() for _ in range(n_layers)]
    misses = prefetch = accesses = 0
    for r, prof in zip(reqs, profiles):
        if policy.startswith("prefetch:"):
            p = prof[policy.split(":", 1)[1]]
            if p.sum() > 0:
                for layer in range(n_layers):
                    cache = caches[layer]
                    for e in np.argsort(-p[layer], kind="stable")[:c][::-1]:
                        e = int(e)
                        if e in cache:
                            cache.move_to_end(e)
                        else:
                            prefetch += 1
                            cache[e] = None
                            if len(cache) > c:
                                cache.popitem(last=False)
        for tok in r.decode:
            for layer in range(n_layers):
                cache = caches[layer]
                for e in tok[layer]:
                    e = int(e)
                    accesses += 1
                    if e in cache:
                        cache.move_to_end(e)
                    else:
                        misses += 1
                        cache[e] = None
                        if len(cache) > c:
                            cache.popitem(last=False)
    return misses, prefetch, accesses


def belady(reqs: list[Req], c: int) -> tuple[int, int]:
    """Optimal offline eviction per layer over the concatenated decode stream."""
    stream = np.concatenate([r.decode for r in reqs if len(r.decode)], axis=0)
    n_tok, n_layers, k = stream.shape
    misses = 0
    for layer in range(n_layers):
        seq = stream[:, layer, :]
        next_use: dict[int, list[int]] = {}
        for t in range(n_tok - 1, -1, -1):
            for e in seq[t]:
                next_use.setdefault(int(e), []).append(t)
        cache: set[int] = set()
        for t in range(n_tok):
            needed = {int(e) for e in seq[t]}
            for e in needed:
                next_use[e].pop()
            for e in needed:
                if e in cache:
                    continue
                misses += 1
                if len(cache) >= c:
                    victims = cache - needed
                    victim = max(
                        victims,
                        key=lambda x: next_use[x][-1] if next_use[x] else 1 << 60,
                    )
                    cache.discard(victim)
                cache.add(e)
    return misses, n_tok * n_layers * k


def main() -> None:
    """Entry point."""
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("prefix", type=Path, help="moe_trace --out prefix")
    ap.add_argument("--min-lcp", type=int, default=32)
    ap.add_argument("--slots", default="8,16,32,48,64")
    ap.add_argument("--no-belady", action="store_true")
    args = ap.parse_args()

    meta, reqs = load(args.prefix)
    n_expert, k = meta["n_expert"], meta["k"]
    slots = [int(s) for s in args.slots.split(",")]
    profiles = build_profiles(reqs, n_expert, args.min_lcp)

    n_dec = sum(len(r.decode) for r in reqs)
    hit = sum(1 for p in profiles if p["cached"])
    print(
        f"model: {meta['desc']}  experts={n_expert} k={k} layers={len(meta['moe_layers'])}"
    )
    print(
        f"requests={len(reqs)} decode_tokens={n_dec} radix_hits={hit} (min_lcp={args.min_lcp})"
    )
    print(
        f"mean cached prefix share: {np.mean([p['cached'] / r.n_prompt for p, r in zip(profiles, reqs)]):.1%}"
    )

    names = ["global", "node", "delta", "node+delta", "full_prefill"]
    print("\n## recall@C of decode routing (requests with a radix hit)")
    print("| C | " + " | ".join(names) + " |")
    print("|---" * (len(names) + 1) + "|")
    for c in slots:
        cells = []
        for name in names:
            vals = [
                recall_at(p[name], r.decode, c)
                for p, r in zip(profiles, reqs)
                if p["cached"]
            ]
            cells.append(f"{np.nanmean(vals):.3f}")
        print(f"| {c} | " + " | ".join(cells) + " |")

    print("\n## cache simulation: expert loads per decode token (demand + prefetch)")
    policies = [
        "lru",
        "prefetch:global",
        "prefetch:node",
        "prefetch:node+delta",
        "prefetch:full_prefill",
    ]
    print(
        "| C | "
        + " | ".join(policies)
        + (" | belady |" if not args.no_belady else " |")
    )
    print("|---" * (len(policies) + 1 + (not args.no_belady)) + "|")
    for c in slots:
        cells = []
        for pol in policies:
            m, pf, _ = simulate(reqs, profiles, c, pol)
            cells.append(f"{(m + pf) / n_dec:.2f} ({m / n_dec:.2f}+{pf / n_dec:.2f})")
        if not args.no_belady:
            bm, _ = belady(reqs, c)
            cells.append(f"{bm / n_dec:.2f}")
        print(f"| {c} | " + " | ".join(cells) + " |")

    print(
        f"\n## per-family routing overlap (Jaccard of top-{n_expert // 4} decode experts, mean over layers)"
    )
    fams = sorted({r.family for r in reqs})
    fam_h = {
        f: normalise(
            np.sum(
                [histogram(r.decode, n_expert) for r in reqs if r.family == f], axis=0
            )
        )
        for f in fams
    }
    top = {
        f: [
            set(np.argsort(-fam_h[f][layer])[: n_expert // 4].tolist())
            for layer in range(fam_h[f].shape[0])
        ]
        for f in fams
    }
    print("| | " + " | ".join(fams) + " |")
    print("|---" * (len(fams) + 1) + "|")
    for a in fams:
        row = [
            f"{np.mean([len(x & y) / len(x | y) for x, y in zip(top[a], top[b])]):.2f}"
            for b in fams
        ]
        print(f"| {a} | " + " | ".join(row) + " |")


if __name__ == "__main__":
    main()
