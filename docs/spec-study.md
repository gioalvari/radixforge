# Cross-request speculative drafting study

## Question and method

This measurement asks whether a draft source shared across completed requests accepts materially more greedy tokens than request-local n-gram drafting. It uses Qwen2.5-7B-Instruct Q4_K_M with llama.cpp's `llama-server`, sequential non-streaming requests, temperature 0, seed 20260925, `--parallel 1 -ngl 99 --ctx-size 16384`, and fresh servers for each mode. Raw outputs, responses, rendered prompts, and tokenizer IDs are in `/tmp/spec_study/`.

The deterministic 100-request workload interleaves 25 requests each from: JSON tool-agent turns with accumulated history; complete Python-file edits using supplied local source files; short energy market paragraph rewrites; and eight-agent regional reports with a shared Markdown/JSON structure. The interleaving makes the global source include unrelated traffic as well as earlier like-family traffic.

## Timing environment

`pmset -g batt` captured while timing:

```text
Now drawing from 'Battery Power'
 -InternalBattery-0 (id=22675555)	90%; discharging; 3:17 remaining present: true
```

The machine was on battery; absolute timings are indicative. Acceptance and the offline lossless simulation depend only on the token streams.

## llama-server results

Decode rate uses `timings.predicted_n / timings.predicted_ms`; speedup is against `none` in that family. Draft acceptance is `draft_n_accepted / draft_n` when the server exposes both fields.

| Family | Config | Decode tok/s | Speedup | Draft acceptance |
| --- | --- | --- | --- | --- |
| tool_json | none | 47.54 | 1.00x | not exposed |
| tool_json | ngram_mod | 49.45 | 1.04x | 36.2% |
| tool_json | ngram_simple | 48.25 | 1.02x | 20.8% |
| tool_json | ngram_cache | 26.40 | 0.56x | 16.6% |
| code_edit | none | 47.34 | 1.00x | not exposed |
| code_edit | ngram_mod | 143.25 | 3.03x | 91.9% |
| code_edit | ngram_simple | 171.82 | 3.63x | 92.3% |
| code_edit | ngram_cache | 34.74 | 0.73x | 3.8% |
| doc_rewrite | none | 48.92 | 1.00x | not exposed |
| doc_rewrite | ngram_mod | 107.07 | 2.19x | 62.6% |
| doc_rewrite | ngram_simple | 131.98 | 2.70x | 81.9% |
| doc_rewrite | ngram_cache | 42.30 | 0.86x | 0.0% |
| agent_reports | none | 46.65 | 1.00x | not exposed |
| agent_reports | ngram_mod | 46.13 | 0.99x | 8.1% |
| agent_reports | ngram_simple | 48.72 | 1.04x | 0.0% |
| agent_reports | ngram_cache | 40.77 | 0.87x | 0.7% |
| overall | none | 47.35 | 1.00x | not exposed |
| overall | ngram_mod | 73.13 | 1.54x | 64.3% |
| overall | ngram_simple | 79.02 | 1.67x | 88.9% |
| overall | ngram_cache | 36.04 | 0.76x | 5.7% |

Output identity: FAIL; 13 mismatches: ngram_mod:agent_reports-006, ngram_mod:agent_reports-008, ngram_mod:tool_json-017, ngram_mod:agent_reports-018, ngram_mod:agent_reports-019, ngram_cache:agent_reports-000, ngram_cache:tool_json-007, ngram_cache:tool_json-011, ngram_cache:agent_reports-011, ngram_cache:agent_reports-013, ngram_cache:agent_reports-018, ngram_cache:agent_reports-019, ngram_cache:agent_reports-024
These greedy mismatches are recorded rather than treated as lossless; they are confined to ngram-mod/cache (ngram-simple matched all baseline outputs in this run).

## Offline exact-match simulation

Each verification step emits one target token plus exactly accepted draft tokens. `local` searches only the current request's prompt and generated output. `global` adds all preceding rendered prompts and outputs in send order; `global_outputs_only` adds only previous outputs. Candidates use longest suffix matches (minimum `m`), favor local ties, choose the most frequent maximal continuation, and are verified against baseline tokens.

| Family | Local tok/step | Global tok/step | Global outputs tok/step | Local accept | Global accept |
| --- | --- | --- | --- | --- | --- |
| tool_json | 1.45 | 2.33 | 2.31 | 9.3% | 16.6% |
| code_edit | 10.21 | 10.35 | 10.48 | 86.2% | 75.6% |
| doc_rewrite | 8.12 | 13.62 | 10.16 | 76.1% | 87.6% |
| agent_reports | 1.20 | 1.65 | 1.62 | 7.8% | 10.7% |
| overall | 2.42 | 3.34 | 3.26 | 35.0% | 30.9% |

All simulated combinations (m ∈ {2, 4}; k ∈ {4, 8, 16, adaptive}) are retained in `/tmp/spec_study/simulation.jsonl`. The table above shows m=2, k=16; it is the highest fixed draft budget and exposes the cross-request ceiling.

## Verification cost and estimated speedup

| Draft k | Target tokens | Forward time (ms) | Relative c(k) |
| --- | --- | --- | --- |
| 0 | 1 | 48.07 | 1.000 |
| 1 | 2 | 61.87 | 1.287 |
| 2 | 3 | 89.04 | 1.852 |
| 4 | 5 | 122.81 | 2.555 |
| 8 | 9 | 123.01 | 2.559 |
| 16 | 17 | 128.26 | 2.668 |

Estimated speedup = simulated tokens per step divided by mean measured/interpolated `c(draft length used)`. This is a target-forward-only estimate; server rates also include existing n-gram implementation overhead.

| Family | Local estimate | Global estimate | Global/local | Best server n-gram |
| --- | --- | --- | --- | --- |
| tool_json | 0.96x | 1.26x | 1.32x | 1.04x |
| code_edit | 4.81x | 4.50x | 0.94x | 3.63x |
| doc_rewrite | 4.10x | 5.41x | 1.32x | 2.70x |
| agent_reports | 0.95x | 1.00x | 1.06x | 1.04x |
| overall | 1.69x | 1.86x | 1.10x | 1.67x |

## Verdict

NO: the global estimate does not clear the stated ≥1.3×-over-best-server threshold in two families.
The qualifying families are: doc_rewrite.
The raw global/local uplift is largest for tool JSON and document rewrites, whereas code editing is already heavily served by local repetition and reports remain low-acceptance under these generic prompts.
The local estimate is a sanity reference against llama-server's local dynamic n-gram measurements; differences reflect llama-server's matching policy, its default n-gram settings, and non-forward overhead.
The cost curve shows that longer verification batches are much cheaper than one forward per accepted token, but their benefit depends on acceptance rather than merely emitting long drafts.
Do not implement cross-request drafting now: only doc_rewrite clears the stated measured-server threshold. Retain request-local n-gram speculation unless a production-like workload clears it in two families within cache-memory limits.
