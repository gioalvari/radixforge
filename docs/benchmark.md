# Shared-prefix server comparison

**Run date:** 2026-11-06

**RadixForge baseline commit:** `a550f2f` plus the metric-attribution change in this branch.

**RadixForge performance-run source:** this branch, after the greedy-sampler change and the explicit Metal synchronization measurement documented below.

**Vendored llama.cpp:** v0.8.0 (`8642541b255bfb5dc39cd351a178be09013a55d6`)

**Model:** `Qwen2.5-0.5B-Instruct-Q4_K_M.gguf`

**Hardware:** Apple M4 Pro; 51,539,607,552 bytes of memory; macOS Metal backend.

This is a single-machine, small-model comparison. It is a reproducible engineering measurement, not a general performance claim.

## Results on llama.cpp v0.8.0

Each cell is the median of three fresh-process repetitions. The bracket is the min--max range across repetitions. Every repetition measured 24 streaming `/completion` requests after one excluded warm-up request.

| Concurrency | Server and cache mode | TTFT p50 (ms) | TTFT p95 (ms) | E2E p50 (ms) | E2E p95 (ms) | Throughput (req/s) | Cache-hit token ratio |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | RadixForge, `--no-prefix-cache` | 44.2 [43.9--44.2] | 45.5 [44.7--45.7] | 107.7 [106.7--107.9] | 109.6 [108.5--110.8] | 9.3 [9.3--9.4] | 0.0% |
| 1 | RadixForge, cache enabled | 8.4 [8.3--8.4] | 9.3 [9.3--9.6] | 70.7 [70.2--71.3] | 73.3 [73.1--75.5] | 14.0 [12.5--14.3] | 96.3% |
| 1 | llama-server, `cache_prompt: false` | 39.3 [39.2--39.4] | 39.7 [39.4--40.1] | 73.8 [73.5--75.2] | 77.3 [77.3--80.1] | 13.4 [13.2--13.5] | 0.0% |
| 1 | llama-server, `cache_prompt: true` | 7.7 [7.7--7.8] | 8.4 [8.3--9.0] | 41.8 [41.7--43.4] | 46.4 [45.8--46.4] | 23.3 [23.2--23.3] | 96.3% |
| 4 | RadixForge, `--no-prefix-cache` | 112.5 [112.4--112.7] | 151.6 [151.5--351.5] | 220.3 [219.6--220.3] | 226.2 [225.7--568.5] | 17.7 [14.1--17.8] | 0.0% |
| 4 | RadixForge, cache enabled | 20.6 [20.6--21.0] | 21.6 [21.5--21.8] | 92.9 [91.7--93.1] | 95.0 [94.4--98.1] | 42.8 [42.6--42.8] | 96.3% |
| 4 | llama-server, `cache_prompt: false` | 140.2 [139.8--140.9] | 149.8 [149.1--150.3] | 186.2 [184.2--188.4] | 197.5 [197.5--198.0] | 21.2 [21.1--21.3] | 0.0% |
| 4 | llama-server, `cache_prompt: true` | 19.6 [19.5--20.0] | 116.8 [115.7--119.4] | 64.9 [64.2--67.3] | 164.1 [162.6--167.2] | 49.2 [47.6--49.4] | 84.2% |

Cache-hit ratio is cache-hit tokens divided by cache-hit plus newly evaluated prompt tokens. For RadixForge it comes from the delta of `/metrics` counters. For llama-server it is aggregated from terminal streaming `timings.cache_n` and `timings.prompt_n` fields. The two counters are comparable as an indication of prompt work avoided, but not guaranteed to have identical internal definitions.

## Component timing snapshot

**Run date:** 2026-11-06. Each value is the median of three fresh RadixForge cache-enabled runs after one excluded warm-up request; each run has 24 streaming requests and uses `--debug-metrics`. Values are `/metrics` counter deltas. Decode time is apportioned between fused prefill and generation tokens by their share of each fused `llama_decode` batch.

| Concurrency | Queue wait (ms/req) | Cache lookup / insert (ms/call) | Prefill decode (ms/token) | Generation decode (ms/token) | Sync (ms/call) | Sampler (ms/call) | Callback (ms/call) | SSE write (ms/write) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.005 | 0.001 / 0.015 | 0.116 | 0.541 | 3.650 | 0.080 | 0.066 | 0.042 |
| 4 | 5.453 | 0.000 / 0.013 | 0.057 | 0.147 | 1.133 | 0.074 | 0.037 | 0.019 |

The former sampler number included asynchronous Metal completion. In debug-metrics mode only, RadixForge now calls and measures `llama_synchronize(ctx)` immediately before sampling; the sampler counter starts afterward. At c=1, 360 generated tokens accumulated 194.8 ms in generation decode, 1,401.5 ms waiting for synchronization, and 30.6 ms in sampler execution. At c=4, the corresponding values were 52.9 ms, 435.0 ms, and 28.5 ms. The 131.0 ms total queue wait at c=4 reflects intentional concurrent request arrival and batching, rather than a per-token cost. These counters are opt-in: without `--debug-metrics`, RadixForge does not allocate the counter set, read timing clocks, or add a synchronization call on these paths.

## Performance fixes

The earlier cache-on concurrency-1 result for RadixForge was 16.8 ms TTFT, 126.2 ms E2E p50, and 7.79 req/s; the post-first-token portion was much slower than llama-server despite similar TTFT. A three-second `sample` capture during a long streaming request showed `llama_sampler_sample` occupying the worker thread, including `ggml_backend_sched_synchronize` / Metal `waitUntilCompleted`, followed by substantial `llama_sampler_top_p_apply` work. The server created a temperature sampler, top-p sampler, and distribution sampler even for `temperature: 0` requests.

RadixForge now creates llama.cpp's greedy sampler directly when `temperature <= 0`, avoiding the full-vocabulary temperature/top-p/distribution sampling path while preserving greedy output. On the same c=1 cache-on workload, E2E p50 fell from 126.2 ms to 67.8 ms and throughput rose from 7.79 to 14.48 req/s; TTFT fell from 16.8 ms to 12.3 ms. The byte output for a representative 24-token Qwen completion was identical before and after the change. Cache-on c=4 throughput rose from 17.80 to 30.29 req/s.

A remaining cache-on E2E gap on v0.8.0 is 70.7 ms versus llama-server's 41.8 ms at c=1, and 92.9 ms versus 64.9 ms at c=4. The split counter shows that the old 3.606 ms/token "sampler" reading at c=1 was almost entirely GPU completion: 3.650 ms/token is synchronization and only 0.080 ms/token is sampling. The remaining difference also includes separate worker scheduling, per-token callback/SSE handling, and different batching/slot implementations; this run does not attribute the gap to a single component.

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
--ctx-size 8192 --parallel 32 --n-gpu-layers 99 --flash-attn auto --no-webui
```

The request body sets `cache_prompt` to `true` or `false`. `--cache-reuse` was left at its documented default of `0`: this comparison measures the server's ordinary per-slot prompt reuse, not KV-shifting chunk reuse. RadixForge has no flash-attention switch and uses llama.cpp context defaults; llama-server explicitly used its default `auto` mode, so flash-attention behavior is not a controlled variable beyond each server's defaults.

## Interpretation

Both servers reduce first-token latency with prompt caching. RadixForge's strict same-prompt cache-on/off comparison reaches 96.3% prompt-token reuse and improves c=1 median TTFT from 39.1 ms to 12.3 ms. After bypassing unnecessary non-greedy sampler stages, RadixForge's cache-enabled c=4 throughput is 30.29 req/s, near llama-server's 37.96 req/s on this workload.

llama-server remains faster for the cache-enabled end-to-end path in this test: 28.9 ms lower E2E p50 at c=1 and 28.0 ms lower at c=4. The strongest evidence-based hypothesis is per-token Metal synchronization/scheduling in RadixForge: its explicit waits total 1,401.5 ms for 384 samples at c=1 and 435.0 ms at c=4, while sampler execution itself is only 30.6 ms and 28.5 ms. Additional measured RadixForge overhead is callback plus SSE work (42.4 ms at c=1; 22.0 ms at c=4), but that alone cannot explain the gap. A remaining unmeasured difference is server architecture: RadixForge's worker/radix-tree scheduling versus llama-server slots and batching. This does not establish a universal winner: llama-server's `cache_prompt` operates through server slots and request scheduling, whereas RadixForge keeps a radix tree that can share a matching prefix across physical sequences, including retained decoded output. The synthetic workload favors one common prefix and does not exercise divergent trees, multi-turn generated-token reuse, eviction pressure, larger models, or mixed arrival patterns.

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
  --out .cache/benchmark-comparison/results.json --repetitions 3
```

## Caveats

- One machine, one 0.5B quantized model, and three repetitions are insufficient for broad performance conclusions.
- Client-observed timing includes local HTTP, scheduler, and stream framing overhead.
- Output length can vary when a model emits EOS before the 16-token limit; throughput is completed requests divided by wall time.
- The harness warms both cache-enabled and cache-disabled runs to reduce model/backend startup effects. Warm-up does not enable reuse when caching is disabled and is excluded from all reported counters and timing samples.
- For older measurements and obsolete implementation details, see the [historical report](history/benchmark-2026.md).
