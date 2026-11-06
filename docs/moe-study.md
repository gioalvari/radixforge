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

### 2026-11-06 representative shared-prefix run

**Conclusion: do not integrate prefix-conditioned expert residency yet.** The
node profile did not predict decode experts better than the global profile, and
the simulated prefetch policy increased total expert loads at every tested
capacity. The workload is deliberately small, so this rules out neither a more
representative production workload nor a policy that accounts for transfer
latency; it does show that routing recall alone is insufficient justification.

The run used `bartowski/OLMoE-1B-7B-0924-Instruct-GGUF` IQ2_M (64 experts,
top-8 routing, 16 MoE layers) on Metal with all layers requested for GPU
offload. The 12-request synthetic workload has three four-request families
(`customer-support`, `data-analysis`, and `software-engineering`) with a
shared system-message prefix and distinct user requests. It generated 372
decode tokens. At `--min-lcp 32`, 9/12 requests had a radix hit and the mean
cached-prefix share was 46.7%.

For requests with a radix hit, recall@16 was 0.496 for the node profile versus
0.497 global, 0.487 delta, 0.520 node+delta, and 0.485 full-prefill reference.
At 32 slots the values were respectively 0.760, 0.769, 0.749, 0.778, and
0.750. Thus the node-specific signal was slightly weaker than global routing;
combining it with the uncached prompt delta improved recall modestly, but not
enough to lower total loads.

In the cache simulation, total expert loads per decode token (demand plus
prefetch) for node prefetch versus LRU were 95.78 versus 93.65 at 8 slots,
64.85 versus 61.41 at 16, 35.31 versus 31.42 at 32, 13.37 versus 10.59 at 48,
and 2.75 versus 2.72 at 64. Node+delta was also worse than LRU at all those
capacities (96.82, 66.49, 36.88, 16.30, and 2.75). There is cache-management
headroom in principle--the Belady oracle was 78.28, 39.55, 14.86, 5.11, and
2.72 loads per decode token--but this prefix profile did not capture it.

The workload and outputs are uncommitted under `/tmp/radixforge-moe/`:

- workload SHA-256:
  `79d9d1640e64bb3d406392efc990bf5fec705e29a909c1da4a673e4f45ca651d`;
- trace binary SHA-256:
  `1d982787c1947a5e45df08026d25f0ef6b56b396fdf134c37b06198b3a2c1e25`;
- trace rows SHA-256:
  `12542daf92654efdac2f44186bdeaa72e9ef812c51f308fd74168fcc2712a1a7`.

The downloaded model is intentionally outside the repository at
`~/.cache/radixforge/models/OLMoE-1B-7B-0924-Instruct-IQ2_M.gguf` (SHA-256
`0e87e90113c6581c0253e176d45ade91a3b92cd1d413f34ffedb2c1e97ed0cdb`).

Even if a profile predicts routing well, this study does not implement expert residency, prefetching, or cache policy changes in RadixForge. Any integration would need to account for model/backend support, transfer costs, cache capacity, and the difference between offline routing recall and end-to-end inference latency.
