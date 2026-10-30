# Current shared-prefix benchmark

**Run date:** 2026-10-30

**Server commit:** `11b07e5f70003fde7087a8886db197ce7ecd11eb`

**Model:** `Qwen2.5-0.5B-Instruct-Q4_K_M.gguf`
**Hardware:** Apple M4 Pro; 51,539,607,552 bytes of memory; macOS Metal backend (`-ngl 99`)

This is a fresh, single-machine measurement of the current hardened server. It is intended to make the current shared-prefix behavior reproducible, not to predict performance for other models, prompts, backends, or hardware.

## Summary

The workload sends 24 streaming chat requests with a repeated synthetic system prompt and 16 maximum generated tokens. It runs each condition in a newly started server process at concurrency 1 and 4.

| Concurrency | Condition | TTFT p50 / p95 (ms) | End-to-end p50 / p95 (ms) | Throughput (req/s) | Cache-hit token ratio |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | Low-reuse control | 75.2 / 78.8 | 149.0 / 171.2 | 6.62 | 1.6% |
| 1 | Warm shared prefix | 14.6 / 15.5 | 87.6 / 110.1 | 11.09 | 97.3% |
| 4 | Low-reuse control | 235.4 / 286.1 | 420.6 / 434.9 | 9.26 | 1.6% |
| 4 | Warm shared prefix | 31.9 / 33.7 | 185.6 / 195.9 | 21.16 | 97.3% |

For the warm condition, the 24 measured requests accounted for 9,302 prompt tokens and 9,053 cache-hit tokens. The control recorded 9,436 prompt tokens and 152 cache-hit tokens.

The low-reuse control is not a cache-disable mode: the current server has no cache-disable CLI flag. Instead, it prepends a distinct request identity before the otherwise shared system prompt. The residual 1.6% cache-hit ratio reflects shared leading tokens before that identity and is reported rather than rounded to zero. The warm condition performs one unmeasured warm-up request first, then uses the shared system prompt for every measured request.

## Method

The historical report used `scripts/ab_open_loop.py` to gather streaming TTFT and completion percentiles and compared fresh server processes. This run keeps that approach—streaming requests, fresh processes, fixed concurrency, and percentile reporting—but uses the current `/metrics` endpoint for cache accounting. It uses a deterministic synthetic prompt instead of downloading HotpotQA so that the benchmark has no network or dataset dependency.

The committed harness is [`scripts/benchmark_current.py`](../scripts/benchmark_current.py). It uses a fixed 24-request workload, a repeated reference paragraph, `temperature: 0`, `max_tokens: 16`, and sequential batches of either one or four concurrent requests. Server settings were:

```text
--ctx-size 8192 --batch-size 512 --max-seq 32 --admit-window-ms 0 \
--request-timeout 120 --kv-high-watermark 0.9 -ngl 99
```

The cache-hit token ratio is the change in `radixforge_cache_hit_tokens_total` divided by the change in `radixforge_prompt_tokens_total` over a measured phase. It is mapper-tracked valid-KV accounting, not a direct llama.cpp occupancy reading.

## Reproduce

Available local models at the time of this run were Qwen2.5 0.5B and 7B Instruct Q4_K_M plus Meta-Llama-3.1-8B-Instruct Q4_K_M. The 0.5B Qwen model was selected to match the model named in the historical report and keep this run short.

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DRADIXFORGE_BUILD_TESTS=ON
cmake --build build -j 10

uvx ruff check scripts/
uvx ruff format --check scripts/

python3 scripts/benchmark_current.py \
  --model ~/local-llm-bench/models/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf \
  --out .cache/benchmark/results.json
```

The harness writes the aggregate JSON report and one server log per condition beneath the supplied output directory. It does not commit model files, logs, or generated results.

## Caveats

- This is one run on one machine with a small quantized model. It has no repetitions, confidence intervals, thermal-control protocol, or cross-model comparison.
- The workload intentionally maximizes prefix overlap in the warm condition. It should not be read as a representative mixed-prompt service benchmark.
- TTFT is measured by the client when it receives the first non-empty SSE `delta.content`, so it includes local HTTP and scheduling overhead.
- Throughput is request count divided by workload wall time; output lengths can vary because the model may stop before 16 tokens.
- The low-reuse control still has a small observed cache hit, as described above; it is a practical cold-ish comparison rather than a strict no-cache baseline.
- For older measurements and obsolete implementation details, see the [historical report](history/benchmark-2026.md).
