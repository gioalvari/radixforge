#!/usr/bin/env python3
"""Simulate lossless n-gram drafting from baseline token streams."""

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import DefaultDict, Dict, Iterable, List, Sequence, Tuple


RAW_DIR = Path("/tmp/spec_study")
BASELINE = RAW_DIR / "server_none.jsonl"
OUT = RAW_DIR / "simulation.jsonl"
MODES = ("local", "global", "global_outputs_only")


def chunks_for_mode(
    mode: str,
    prior: Sequence[Dict[str, object]],
    prompt: List[int],
    generated: List[int],
) -> List[List[int]]:
    """Return the draft-source streams available before the next target step."""
    local = prompt + generated
    if mode == "local":
        return [local]
    previous: List[int] = []
    for record in prior:
        if mode == "global":
            previous.extend(
                list(record["prompt_tokens"]) + list(record["output_tokens"])
            )
        else:
            previous.extend(list(record["output_tokens"]))
    # Treat all past traffic as one chronological source, as a shared cache would;
    # retaining the local stream separately permits explicit local tie preference.
    return [previous, local]


def candidates(
    chunks: Sequence[List[int]],
    context: List[int],
    m: int,
    maximum: int,
) -> List[Tuple[int, int, int, Tuple[int, ...]]]:
    """Find source continuations matching the longest target-context suffix.

    The return is (match length, source priority, continuation length, tokens).
    Source priority is zero for the local stream, making it win equal-length ties.
    """
    if len(context) < m:
        return []
    suffix = tuple(context[-m:])
    index: DefaultDict[Tuple[int, ...], List[Tuple[int, int]]] = defaultdict(list)
    for source_index, chunk in enumerate(chunks):
        for position in range(len(chunk) - m):
            index[tuple(chunk[position : position + m])].append(
                (source_index, position)
            )
    found: List[Tuple[int, int, int, Tuple[int, ...]]] = []
    for source_index, position in index[suffix]:
        chunk = chunks[source_index]
        before = 0
        while (
            before < len(context) - m
            and position - before - 1 >= 0
            and chunk[position - before - 1] == context[-m - before - 1]
        ):
            before += 1
        match = m + before
        continuation = tuple(chunk[position + m : position + m + maximum])
        if continuation:
            # The appended local stream is always last. Favor it on equal matches.
            priority = 0 if source_index == len(chunks) - 1 else 1
            found.append((match, priority, len(continuation), continuation))
    return found


def choose_draft(
    chunks: Sequence[List[int]], context: List[int], m: int, k: int
) -> List[int]:
    """Select an n-gram draft by longest match, local tie preference, then frequency."""
    found = candidates(chunks, context, m, k)
    if not found:
        return []
    best_match = max(item[0] for item in found)
    found = [item for item in found if item[0] == best_match]
    if any(item[1] == 0 for item in found):
        found = [item for item in found if item[1] == 0]
    alternatives = Counter(item[3] for item in found)
    winner = max(alternatives, key=lambda item: (alternatives[item], item))
    return list(winner)


def simulate_request(
    record: Dict[str, object],
    prior: Sequence[Dict[str, object]],
    mode: str,
    m: int,
    k: int,
    adaptive: bool,
) -> Dict[str, object]:
    """Simulate greedy draft verification against the known lossless target output."""
    prompt = list(record["prompt_tokens"])
    target = list(record["output_tokens"])
    generated: List[int] = []
    steps = drafted = accepted = steps_with_draft = 0
    draft_lengths: List[int] = []
    last_accepted = 0
    while len(generated) < len(target):
        current_k = min(16, max(1, 2 * last_accepted + 1)) if adaptive else k
        source = chunks_for_mode(mode, prior, prompt, generated)
        proposed = choose_draft(source, prompt + generated, m, current_k)
        draft_lengths.append(len(proposed))
        if proposed:
            steps_with_draft += 1
            drafted += len(proposed)
        exact = 0
        for proposed_token, target_token in zip(proposed, target[len(generated) :]):
            if proposed_token != target_token:
                break
            exact += 1
        accepted += exact
        generated.extend(target[len(generated) : len(generated) + exact + 1])
        last_accepted = exact
        steps += 1
    return {
        "id": record["id"],
        "family": record["family"],
        "mode": mode,
        "m": m,
        "k": "adaptive" if adaptive else k,
        "output_tokens": len(target),
        "steps": steps,
        "drafted": drafted,
        "accepted": accepted,
        "steps_with_draft": steps_with_draft,
        "tokens_per_step": len(target) / steps if steps else 0.0,
        "fraction_steps_with_draft": steps_with_draft / steps if steps else 0.0,
        "acceptance_rate": accepted / drafted if drafted else 0.0,
        "mean_draft_length": statistics.mean(draft_lengths) if draft_lengths else 0.0,
        "draft_length_counts": dict(Counter(draft_lengths)),
    }


def main() -> None:
    """Run all requested source and lookback configurations."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, default=BASELINE)
    parser.add_argument("--out", type=Path, default=OUT)
    arguments = parser.parse_args()
    records = [
        json.loads(line)
        for line in arguments.baseline.read_text(encoding="utf-8").splitlines()
        if line
    ]
    results: List[Dict[str, object]] = []
    for mode in MODES:
        for m in (2, 4):
            for k, adaptive in ((4, False), (8, False), (16, False), (0, True)):
                prior: List[Dict[str, object]] = []
                for record in records:
                    results.append(
                        simulate_request(record, prior, mode, m, k, adaptive)
                    )
                    prior.append(record)
    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    with arguments.out.open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(result, sort_keys=True) + "\n")
    print(f"wrote {len(results)} simulations to {arguments.out}")


if __name__ == "__main__":
    main()
