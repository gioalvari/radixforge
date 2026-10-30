# MoE expert-routing residency study

**Status:** research tooling only; not integrated into the RadixForge server.

## Question

Prefix caching removes repeated prompt prefill work, but it also removes an observable signal: the experts that a shared prompt routed through. This study asks whether a radix-tree prefix can provide a useful expert-residency profile for the decode phase of later requests that match that prefix.

The offline comparison is between:

- a global profile from all earlier decode traffic;
- a profile from earlier requests sharing a sufficiently long token prefix (the profile a radix node could retain);
- the request's uncached prompt delta;
- a combined node-plus-delta profile; and
- a full-prefill profile as a reference that is unavailable once the prefix is reused.

It evaluates recall of the experts selected during decode and simulates a fixed-capacity, per-layer expert cache under LRU, profile-prefetch-plus-LRU, and optional Belady-oracle policies.

## Trace collection

`tools/moe_trace.cpp` is a standalone llama.cpp tracing executable built as `build/tools/moe_trace` when `RADIXFORGE_BUILD_TOOLS=ON` (the default). It requires a mixture-of-experts GGUF with an embedded chat template and an `expert_used_count` model metadata value.

For every JSONL request, the tool:

1. renders the model chat template and tokenizes the prompt;
2. clears the llama.cpp KV cache and greedily generates up to the request limit;
3. captures each `ffn_moe_argsort-<layer>` graph tensor through llama.cpp's evaluation callback; and
4. records the first `k` ranked experts per routed token and MoE layer.

The input is one JSON object per line. Each object needs `id` and `messages`; optional fields are `family`, `agent`, `turn`, and `max_tokens`.

```json
{"id":"request-001","family":"example","messages":[{"role":"user","content":"Hello"}],"max_tokens":32}
```

Build and collect a trace:

```bash
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release
cmake --build build -j 5 --target moe_trace

./build/tools/moe_trace \
  -m path/to/moe-model.gguf \
  --data workload.jsonl \
  --out results/moe-trace \
  --limit 100 \
  --max-tokens 64 \
  -c 8192 \
  -ngl 99
```

`--limit` is optional (`0` means all input rows); `--max-tokens` overrides per-row limits; `-c` sets context size; and `-ngl` sets llama.cpp GPU-layer offload.

The collector writes three related files for an output prefix:

| File | Contents |
| --- | --- |
| `PREFIX.meta.json` | Model description, expert count, selected-expert count `k`, MoE layer IDs, and binary layout. |
| `PREFIX.bin` | `uint8` expert IDs in `[n_routed][n_moe_layers][k]` order. |
| `PREFIX.jsonl` | Per-request IDs, workload fields, prompt/generated token counts, byte offset into the binary trace, timings, token IDs, and generated text. |

The trace is deliberately independent of the server: it starts every request with an empty KV cache and records model routing rather than changing model execution.

## Offline analysis

`scripts/moe_study/analyze.py` reads the three files above. It is a standalone Python 3.11+ script with a NumPy dependency declared in its script metadata.

```bash
uv run scripts/moe_study/analyze.py results/moe-trace \
  --min-lcp 32 \
  --slots 8,16,32,48,64
```

`--min-lcp` is the minimum shared token-prefix length for a request to count as a radix hit. `--slots` sets the number of expert-residency slots per MoE layer. Add `--no-belady` to skip the oracle simulation.

The report includes:

- request count, decode-token count, radix-hit count, and mean cached-prefix share;
- recall@C for each profile over decode routing, restricted to requests with a radix hit;
- demand misses and prefetch loads per decode token for LRU and profile-prefetch policies, with optional Belady comparison; and
- pairwise family overlap using the Jaccard similarity of high-frequency decode experts.

## Findings and scope

No generated MoE trace or analysis output is committed in this repository, so this document intentionally reports no numerical finding. The tool and analysis script establish a reproducible measurement path; a conclusion requires an MoE model, representative JSONL workload, and saved trace output.

Even if a profile predicts routing well, this study does not implement expert residency, prefetching, or cache policy changes in RadixForge. Any integration would need to account for model/backend support, transfer costs, cache capacity, and the difference between offline routing recall and end-to-end inference latency.
