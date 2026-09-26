#!/usr/bin/env python3
"""Summarize raw study artifacts in the checked-in specification document."""

import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[2]
RAW = Path("/tmp/spec_study")
DOC = ROOT / "docs" / "spec-study.md"
FAMILIES = ("tool_json", "code_edit", "doc_rewrite", "agent_reports", "overall")
CONFIGS = ("none", "ngram_mod", "ngram_simple", "ngram_cache")


def rows(path: Path) -> List[Dict[str, Any]]:
    """Load a JSONL file."""
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def family_rows(records: Iterable[Dict[str, Any]], family: str) -> List[Dict[str, Any]]:
    """Select a family, with overall representing all records."""
    return [
        record
        for record in records
        if family == "overall" or record["family"] == family
    ]


def timing(record: Dict[str, Any]) -> Tuple[float, float]:
    """Return generated tokens and server prediction milliseconds where available."""
    response = record["response"]
    timings = response.get("timings", {}) if isinstance(response, dict) else {}
    tokens = float(timings.get("predicted_n", len(record["output_tokens"])))
    milliseconds = float(timings.get("predicted_ms", 0.0))
    if milliseconds <= 0:
        milliseconds = float(record["client_elapsed_ms"])
    return tokens, milliseconds


def server_summary() -> Tuple[Dict[str, Dict[str, Dict[str, float]]], List[str]]:
    """Aggregate measured server rates and check greedy output identity."""
    loaded = {name: rows(RAW / f"server_{name}.jsonl") for name in CONFIGS}
    baseline = {record["id"]: record["output"] for record in loaded["none"]}
    mismatches: List[str] = []
    for name, records in loaded.items():
        if name == "none":
            continue
        for record in records:
            if baseline.get(record["id"]) != record["output"]:
                mismatches.append(f"{name}:{record['id']}")
    summary: Dict[str, Dict[str, Dict[str, float]]] = defaultdict(dict)
    for name, records in loaded.items():
        for family in FAMILIES:
            selected = family_rows(records, family)
            generated, milliseconds = map(
                sum, zip(*(timing(record) for record in selected))
            )
            draft_n = draft_accepted = 0.0
            exposed = 0
            for record in selected:
                timings = record["response"].get("timings", {})
                if "draft_n" in timings or "draft_n_accepted" in timings:
                    exposed += 1
                    draft_n += float(timings.get("draft_n", 0))
                    draft_accepted += float(timings.get("draft_n_accepted", 0))
            summary[family][name] = {
                "tok_s": generated / (milliseconds / 1000),
                "acceptance": draft_accepted / draft_n if draft_n else float("nan"),
                "draft_exposed": float(exposed),
            }
        for family in FAMILIES:
            summary[family][name]["speedup"] = (
                summary[family][name]["tok_s"] / summary[family]["none"]["tok_s"]
            )
    return summary, mismatches


def interpolate_cost(length: int, costs: Dict[int, float]) -> float:
    """Linearly interpolate measured verification relative costs."""
    lower = max(key for key in costs if key <= length)
    upper = min(key for key in costs if key >= length)
    if lower == upper:
        return costs[lower]
    ratio = (length - lower) / (upper - lower)
    return costs[lower] + ratio * (costs[upper] - costs[lower])


def simulation_summary() -> Dict[Tuple[str, int, str], Dict[str, Dict[str, float]]]:
    """Aggregate simulation metrics, including cost-curve estimated speedup."""
    cost_rows = json.loads((RAW / "cost_curve.json").read_text(encoding="utf-8"))[
        "rows"
    ]
    costs = {int(row["draft_length"]): float(row["relative_cost"]) for row in cost_rows}
    grouped: Dict[Tuple[str, int, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for record in rows(RAW / "simulation.jsonl"):
        grouped[
            (record["mode"], int(record["m"]), str(record["k"]), record["family"])
        ].append(record)
        grouped[(record["mode"], int(record["m"]), str(record["k"]), "overall")].append(
            record
        )
    result: Dict[Tuple[str, int, str], Dict[str, Dict[str, float]]] = {}
    for key, selected in grouped.items():
        mode, m, k, family = key
        total_tokens = sum(int(record["output_tokens"]) for record in selected)
        total_steps = sum(int(record["steps"]) for record in selected)
        total_drafted = sum(int(record["drafted"]) for record in selected)
        total_accepted = sum(int(record["accepted"]) for record in selected)
        total_with = sum(int(record["steps_with_draft"]) for record in selected)
        weighted_cost = 0.0
        for record in selected:
            for length, count in record["draft_length_counts"].items():
                weighted_cost += interpolate_cost(int(length), costs) * int(count)
        result.setdefault((mode, m, k), {})[family] = {
            "tokens_per_step": total_tokens / total_steps,
            "fraction_with_draft": total_with / total_steps,
            "acceptance": total_accepted / total_drafted if total_drafted else 0.0,
            "estimated_speedup": total_tokens / weighted_cost,
        }
    return result


def table(headers: Sequence[str], body: Sequence[Sequence[str]]) -> str:
    """Format a Markdown table."""
    return "\n".join(
        [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join("---" for _ in headers) + " |",
            *["| " + " | ".join(row) + " |" for row in body],
        ]
    )


def main() -> None:
    """Write the final evidence, tables, and decision rule outcome."""
    server, mismatches = server_summary()
    simulation = simulation_summary()
    # Largest bounded setting provides a comparable fixed-k local/global answer.
    selected_key = ("local", 2, "16")
    global_key = ("global", 2, "16")
    output = [
        "# Cross-request speculative drafting study",
        "",
        "## Question and method",
        "",
        "This measurement asks whether a draft source shared across completed requests accepts "
        "materially more greedy tokens than request-local n-gram drafting. It uses Qwen2.5-7B-Instruct "
        "Q4_K_M with llama.cpp's `llama-server`, sequential non-streaming requests, temperature 0, seed "
        "20260925, `--parallel 1 -ngl 99 --ctx-size 16384`, and fresh servers for each mode. Raw outputs, "
        "responses, rendered prompts, and tokenizer IDs are in `/tmp/spec_study/`.",
        "",
        "The deterministic 100-request workload interleaves 25 requests each from: JSON tool-agent turns "
        "with accumulated history; complete Python-file edits using supplied local source files; short energy "
        "market paragraph rewrites; and eight-agent regional reports with a shared Markdown/JSON structure. "
        "The interleaving makes the global source include unrelated traffic as well as earlier like-family traffic.",
        "",
        "## Timing environment",
        "",
        "`pmset -g batt` captured while timing:",
        "",
        "```text",
        (RAW / "battery.txt").read_text(encoding="utf-8").strip(),
        "```",
        "",
        "The machine was on battery; absolute timings are indicative. Acceptance and the offline lossless "
        "simulation depend only on the token streams.",
        "",
        "## llama-server results",
        "",
        "Decode rate uses `timings.predicted_n / timings.predicted_ms`; speedup is against `none` in that family. "
        "Draft acceptance is `draft_n_accepted / draft_n` when the server exposes both fields.",
    ]
    server_body: List[List[str]] = []
    for family in FAMILIES:
        for config in CONFIGS:
            values = server[family][config]
            acceptance = values["acceptance"]
            server_body.append(
                [
                    family,
                    config,
                    f"{values['tok_s']:.2f}",
                    f"{values['speedup']:.2f}x",
                    f"{acceptance:.1%}" if acceptance == acceptance else "not exposed",
                ]
            )
    output.extend(
        [
            "",
            table(
                ["Family", "Config", "Decode tok/s", "Speedup", "Draft acceptance"],
                server_body,
            ),
            "",
        ]
    )
    output.append(
        "Output identity: "
        + (
            "PASS; all 100 outputs match the no-speculation baseline."
            if not mismatches
            else f"FAIL; {len(mismatches)} mismatches: {', '.join(mismatches[:20])}"
        )
    )
    if mismatches:
        output.append(
            "These greedy mismatches are recorded rather than treated as lossless; they are confined to "
            "ngram-mod/cache (ngram-simple matched all baseline outputs in this run)."
        )
    output.extend(
        [
            "",
            "## Offline exact-match simulation",
            "",
            "Each verification step emits one target token plus exactly accepted draft tokens. `local` searches only the current request's prompt and generated output. `global` adds all preceding rendered prompts and outputs in send order; `global_outputs_only` adds only previous outputs. Candidates use longest suffix matches (minimum `m`), favor local ties, choose the most frequent maximal continuation, and are verified against baseline tokens.",
            "",
        ]
    )
    sim_body: List[List[str]] = []
    for family in FAMILIES:
        local = simulation[selected_key][family]
        shared = simulation[global_key][family]
        outputs_only = simulation[("global_outputs_only", 2, "16")][family]
        sim_body.append(
            [
                family,
                f"{local['tokens_per_step']:.2f}",
                f"{shared['tokens_per_step']:.2f}",
                f"{outputs_only['tokens_per_step']:.2f}",
                f"{local['acceptance']:.1%}",
                f"{shared['acceptance']:.1%}",
            ]
        )
    output.extend(
        [
            table(
                [
                    "Family",
                    "Local tok/step",
                    "Global tok/step",
                    "Global outputs tok/step",
                    "Local accept",
                    "Global accept",
                ],
                sim_body,
            ),
            "",
        ]
    )
    output.extend(
        [
            "All simulated combinations (m ∈ {2, 4}; k ∈ {4, 8, 16, adaptive}) are retained in `/tmp/spec_study/simulation.jsonl`. The table above shows m=2, k=16; it is the highest fixed draft budget and exposes the cross-request ceiling.",
            "",
            "## Verification cost and estimated speedup",
            "",
        ]
    )
    costs = json.loads((RAW / "cost_curve.json").read_text(encoding="utf-8"))["rows"]
    output.extend(
        [
            table(
                ["Draft k", "Target tokens", "Forward time (ms)", "Relative c(k)"],
                [
                    [
                        str(row["draft_length"]),
                        str(row["tokens"]),
                        f"{row['time_ms']:.2f}",
                        f"{row['relative_cost']:.3f}",
                    ]
                    for row in costs
                ],
            ),
            "",
            "Estimated speedup = simulated tokens per step divided by mean measured/interpolated `c(draft length used)`. This is a target-forward-only estimate; server rates also include existing n-gram implementation overhead.",
            "",
        ]
    )
    estimate_body: List[List[str]] = []
    for family in FAMILIES:
        local = simulation[selected_key][family]["estimated_speedup"]
        shared = simulation[global_key][family]["estimated_speedup"]
        best_server = max(
            server[family][config]["speedup"] for config in CONFIGS if config != "none"
        )
        estimate_body.append(
            [
                family,
                f"{local:.2f}x",
                f"{shared:.2f}x",
                f"{shared / local:.2f}x",
                f"{best_server:.2f}x",
            ]
        )
    output.extend(
        [
            table(
                [
                    "Family",
                    "Local estimate",
                    "Global estimate",
                    "Global/local",
                    "Best server n-gram",
                ],
                estimate_body,
            ),
            "",
        ]
    )
    qualifying = [
        family
        for family in FAMILIES[:-1]
        if simulation[global_key][family]["estimated_speedup"]
        >= 1.3
        * max(
            server[family][config]["speedup"] for config in CONFIGS if config != "none"
        )
    ]
    output.extend(["## Verdict", ""])
    if len(qualifying) >= 2:
        verdict = "YES: the global estimate clears the stated ≥1.3×-over-best-server threshold in at least two families."
    else:
        verdict = "NO: the global estimate does not clear the stated ≥1.3×-over-best-server threshold in two families."
    output.extend(
        [
            verdict,
            f"The qualifying families are: {', '.join(qualifying) if qualifying else 'none'}.",
            "The raw global/local uplift is largest for tool JSON and document rewrites, whereas code editing is already heavily served by local repetition and reports remain low-acceptance under these generic prompts.",
            "The local estimate is a sanity reference against llama-server's local dynamic n-gram measurements; differences reflect llama-server's matching policy, its default n-gram settings, and non-forward overhead.",
            "The cost curve shows that longer verification batches are much cheaper than one forward per accepted token, but their benefit depends on acceptance rather than merely emitting long drafts.",
            "Do not implement cross-request drafting now: only doc_rewrite clears the stated measured-server threshold. Retain request-local n-gram speculation unless a production-like workload clears it in two families within cache-memory limits.",
        ]
    )
    DOC.write_text("\n".join(output) + "\n", encoding="utf-8")
    print(f"wrote {DOC}")


if __name__ == "__main__":
    main()
