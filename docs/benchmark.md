# Shared-prefix server comparison

**Run date:** 2026-11-06

**RadixForge baseline commit:** `a550f2f` plus the metric-attribution change in this branch.

**RadixForge performance-run source:** this branch, after the greedy-sampler change and the explicit Metal synchronization measurement documented below.

**Vendored llama.cpp:** v0.8.0 (`8642541b255bfb5dc39cd351a178be09013a55d6`)

**Model:** `Qwen2.5-0.5B-Instruct-Q4_K_M.gguf`

**Hardware:** Apple M4 Pro; 51,539,607,552 bytes of memory; macOS Metal backend.

This is a single-machine, small-model comparison. It is a reproducible engineering measurement, not a general performance claim.

## Matched-KV results on llama.cpp v0.8.0

Each cell is the median of three fresh-process repetitions. The bracket is the min--max range across repetitions. Every repetition measured 24 streaming `/completion` requests after one excluded warm-up request.

| Concurrency | Server and cache mode | TTFT p50 (ms) | TTFT p95 (ms) | E2E p50 (ms) | E2E p95 (ms) | Throughput (req/s) | Cache-hit token ratio |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | RadixForge, `--no-prefix-cache` | 43.9 [42.7--44.6] | 47.0 [46.0--47.0] | 111.4 [102.7--114.7] | 120.4 [108.3--123.1] | 8.9 [8.4--9.4] | 0.0% |
| 1 | RadixForge, cache enabled | 8.4 [8.3--9.4] | 10.2 [9.8--10.4] | 74.4 [73.4--75.0] | 85.8 [78.8--86.2] | 13.1 [12.8--13.7] | 96.3% |
| 1 | llama-server, `cache_prompt: false`, `--kv-unified` | 44.5 [43.9--44.9] | 48.2 [46.9--48.7] | 113.2 [113.1--114.8] | 125.2 [117.6--126.6] | 7.6 [7.3--8.8] | 0.0% |
| 1 | llama-server, `cache_prompt: true`, `--kv-unified` | 9.8 [8.8--10.2] | 18.6 [12.4--18.6] | 84.3 [79.0--87.0] | 93.0 [83.7--94.2] | 10.2 [9.9--10.3] | 96.3% |
| 4 | RadixForge, `--no-prefix-cache` | 128.4 [121.4--131.4] | 182.8 [179.0--564.7] | 257.2 [257.0--260.5] | 615.0 [266.5--639.3] | 12.5 [12.4--12.7] | 0.0% |
| 4 | RadixForge, cache enabled | 22.6 [21.7--22.9] | 26.9 [26.9--28.4] | 106.1 [104.3--108.8] | 118.5 [114.7--118.9] | 36.9 [35.9--37.8] | 96.3% |
| 4 | llama-server, `cache_prompt: false`, `--kv-unified` | 172.4 [163.7--174.8] | 184.5 [176.3--185.1] | 266.9 [253.3--283.2] | 281.8 [269.2--293.1] | 14.9 [14.0--15.1] | 0.0% |
| 4 | llama-server, `cache_prompt: true`, `--kv-unified` | 21.3 [21.1--21.7] | 128.5 [127.5--129.3] | 107.9 [107.9--112.6] | 211.7 [210.8--212.8] | 31.7 [31.5--31.8] | 84.2% |

Cache-hit ratio is cache-hit tokens divided by cache-hit plus newly evaluated prompt tokens. For RadixForge it comes from the delta of `/metrics` counters. For llama-server it is aggregated from terminal streaming `timings.cache_n` and `timings.prompt_n` fields. The two counters are comparable as an indication of prompt work avoided, but not guaranteed to have identical internal definitions.

## Component timing snapshot

**Run date:** 2026-11-06. Each value is the median of three fresh RadixForge cache-enabled runs after one excluded warm-up request; each run has 24 streaming requests and uses `--debug-metrics`. Values are `/metrics` counter deltas. Decode time is apportioned between fused prefill and generation tokens by their share of each fused `llama_decode` batch.

| Concurrency | Queue wait (ms/req) | Cache lookup / insert (ms/call) | Prefill decode (ms/token) | Generation decode (ms/token) | Sync (ms/call) | Sampler (ms/call) | Callback (ms/call) | SSE write (ms/write) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.007 | 0.001 / 0.016 | 0.121 | 0.615 | 3.845 | 0.089 | 0.074 | 0.047 |
| 4 | 6.108 | 0.000 / 0.014 | 0.061 | 0.159 | 1.356 | 0.081 | 0.041 | 0.021 |

The former sampler number included asynchronous Metal completion. In debug-metrics mode only, RadixForge calls and measures `llama_synchronize(ctx)` immediately before sampling; the sampler counter starts afterward. At c=1, 360 generated tokens accumulated 221.5 ms in generation decode, 1,476.5 ms waiting for synchronization, and 34.3 ms in sampler execution. At c=4, the corresponding values were 57.1 ms, 520.8 ms, and 31.1 ms. The 146.6 ms total queue wait at c=4 reflects intentional concurrent request arrival and batching, rather than a per-token cost. These counters are opt-in: without `--debug-metrics`, RadixForge does not allocate the counter set, read timing clocks, or add a synchronization call on these paths.

## Performance fixes

The earlier cache-on concurrency-1 result for RadixForge was 16.8 ms TTFT, 126.2 ms E2E p50, and 7.79 req/s; the post-first-token portion was much slower than llama-server despite similar TTFT. A three-second `sample` capture during a long streaming request showed `llama_sampler_sample` occupying the worker thread, including `ggml_backend_sched_synchronize` / Metal `waitUntilCompleted`, followed by substantial `llama_sampler_top_p_apply` work. The server created a temperature sampler, top-p sampler, and distribution sampler even for `temperature: 0` requests.

RadixForge now creates llama.cpp's greedy sampler directly when `temperature <= 0`, avoiding the full-vocabulary temperature/top-p/distribution sampling path while preserving greedy output. On the same c=1 cache-on workload, E2E p50 fell from 126.2 ms to 67.8 ms and throughput rose from 7.79 to 14.48 req/s; TTFT fell from 16.8 ms to 12.3 ms. The byte output for a representative 24-token Qwen completion was identical before and after the change. Cache-on c=4 throughput rose from 17.80 to 30.29 req/s.

The prior comparison was not KV-equivalent. With `--parallel 32`, llama-server defaults to `kv_unified = false`, which gives each slot a 256-token KV stream; RadixForge requires one unified 8192-cell pool for zero-copy radix-prefix sharing. Running llama-server with `--kv-unified` makes the c=1 cache-on E2E p50 84.3 ms, versus 74.4 ms for RadixForge; at c=4 it is 107.9 ms versus 106.1 ms. The split counter shows that the 3.845 ms/token c=1 synchronization time is GPU completion, while sampling itself is only 0.089 ms/token.

## Method

The workload is deterministic and network-independent: 24 raw completion prompts share a repeated synthetic reference paragraph, vary only in a short final question, use `temperature: 0`, and request at most 16 generated tokens. Clients send sequential batches at concurrency 1 or 4 and measure TTFT at the first non-empty SSE content event and end-to-end time at stream completion.

Each condition starts a fresh process, issues one excluded warm-up request, then runs the same 24 prompts. Three repetitions were performed. The committed harness is [`scripts/benchmark_current.py`](../scripts/benchmark_current.py), which writes JSON results and per-process logs beneath its output directory.

RadixForge was started with:

```text
--ctx-size 8192 --batch-size 512 --max-seq 32 --admit-window-ms 0 \
--request-timeout 120 --kv-high-watermark 0.9 -ngl 99
```

The cache-off condition adds `--no-prefix-cache` and otherwise uses exactly the same prompts and settings.

Vendored llama-server was built separately in Release mode with `LLAMA_BUILD_SERVER=ON` and `GGML_METAL=ON`, then started with:

```text
--ctx-size 8192 --parallel 32 --n-gpu-layers 99 --flash-attn auto --no-webui --kv-unified
```

The request body sets `cache_prompt` to `true` or `false`. `--cache-reuse` was left at its documented default of `0`: this comparison measures the server's ordinary per-slot prompt reuse, not KV-shifting chunk reuse. `--kv-unified` is required for a like-for-like shared 8192-cell pool: without it, explicit `--parallel 32` leaves llama-server in its default split-KV mode, with 256 cells per slot. RadixForge has no flash-attention switch and uses llama.cpp context defaults; llama-server explicitly used its default `auto` mode, so flash-attention behavior is not a controlled variable beyond each server's defaults.

## Interpretation

Both servers reduce first-token latency with prompt caching. RadixForge's strict same-prompt cache-on/off comparison reaches 96.3% prompt-token reuse and improves c=1 median TTFT from 43.9 ms to 8.4 ms. Its cache-enabled c=4 throughput is 36.9 req/s, above the 31.7 req/s from unified-KV llama-server on this workload.

The mismatch hypothesis is confirmed. llama.cpp computes unified `n_kv` from the maximum occupied cell index, padded to 256, and the Metal flash-attention kernels iterate over that `n_kv`; the standard kernel skips all-masked blocks, but the vector kernel has no such block bitmap. Split-KV llama-server therefore had a fundamentally smaller 256-cell attention domain. No RadixForge runtime change is safe here: compacting retained cells would require copying KV tensors and updating all radix-tree sequence mappings, defeating its zero-copy shared-prefix invariant. With unified KV matched, RadixForge is 9.9 ms faster at c=1 and 1.8 ms faster at c=4 E2E p50, despite 3.845 ms/token c=1 GPU synchronization plus 0.615 ms/token generation decode. This does not establish a universal winner: llama-server's `cache_prompt` operates through server slots and request scheduling, whereas RadixForge keeps a radix tree that can share a matching prefix across physical sequences, including retained decoded output. The synthetic workload favors one common prefix and does not exercise divergent trees, multi-turn generated-token reuse, eviction pressure, larger models, or mixed arrival patterns.

## Reproduce

Build RadixForge and the vendored upstream server separately:

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j 6

cmake -S vendor/llama.cpp -B build-llama-server \
  -DCMAKE_BUILD_TYPE=Release -DLLAMA_BUILD_SERVER=ON \
  -DGGML_METAL=ON -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF
cmake --build build-llama-server --target llama-server -j 6

uvx ruff check scripts/
uvx ruff format --check scripts/
python3 scripts/benchmark_current.py \
  --model ~/local-llm-bench/models/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf \
  --out .cache/benchmark-comparison/results.json --repetitions 3 \
  --debug-metrics --llama-kv-unified
```

## Caveats

- One machine, one 0.5B quantized model, and three repetitions are insufficient for broad performance conclusions.
- Client-observed timing includes local HTTP, scheduler, and stream framing overhead.
- Output length can vary when a model emits EOS before the 16-token limit; throughput is completed requests divided by wall time.
- The harness warms both cache-enabled and cache-disabled runs to reduce model/backend startup effects. Warm-up does not enable reuse when caching is disabled and is excluded from all reported counters and timing samples.
- For older measurements and obsolete implementation details, see the [historical report](history/benchmark-2026.md).
