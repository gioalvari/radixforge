# Shared-prefix server comparison

**Run date:** 2026-10-30

**RadixForge baseline commit:** `d355896a8688b2e5886411b932fb22cee87c735c`

**RadixForge performance-run source:** this branch, after the greedy-sampler change documented below.

**Vendored llama.cpp commit:** `f728adab683387b2bdb085b3854a0f3487a13d0a`

**Model:** `Qwen2.5-0.5B-Instruct-Q4_K_M.gguf`

**Hardware:** Apple M4 Pro; 51,539,607,552 bytes of memory; macOS Metal backend.

This is a single-machine, small-model comparison. It is a reproducible engineering measurement, not a general performance claim.

## Results after the performance fix

Each cell is the median of three fresh-process repetitions. The bracket is the min--max range across repetitions. Every repetition measured 24 streaming `/completion` requests after one excluded warm-up request.

| Concurrency | Server and cache mode | TTFT p50 (ms) | TTFT p95 (ms) | E2E p50 (ms) | E2E p95 (ms) | Throughput (req/s) | Cache-hit token ratio |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | RadixForge, `--no-prefix-cache` | 39.1 [39.0--39.2] | 39.7 [39.7--39.8] | 94.0 [93.7--94.7] | 98.1 [95.4--99.3] | 10.58 [10.47--10.59] | 0.0% |
| 1 | RadixForge, cache enabled | 12.3 [12.2--12.3] | 13.5 [12.9--15.6] | 67.8 [67.8--67.9] | 76.4 [72.4--135.5] | 14.48 [12.53--14.65] | 96.3% |
| 1 | llama-server, `cache_prompt: false` | 39.3 [39.2--39.4] | 40.2 [40.2--41.6] | 74.5 [73.8--76.2] | 77.1 [76.1--87.9] | 13.52 [12.91--13.57] | 0.0% |
| 1 | llama-server, `cache_prompt: true` | 12.3 [12.1--12.4] | 15.6 [12.7--20.1] | 46.0 [45.4--47.2] | 54.0 [49.3--112.1] | 20.74 [16.48--21.56] | 96.3% |
| 4 | RadixForge, `--no-prefix-cache` | 117.3 [113.6--119.0] | 160.4 [156.4--161.0] | 255.5 [248.5--258.8] | 274.7 [264.0--380.8] | 15.03 [14.18--15.27] | 0.0% |
| 4 | RadixForge, cache enabled | 26.5 [26.0--26.6] | 30.3 [27.3--96.0] | 130.7 [129.6--164.3] | 139.6 [134.0--338.5] | 30.29 [20.01--30.32] | 96.3% |
| 4 | llama-server, `cache_prompt: false` | 148.4 [147.7--150.6] | 157.7 [156.6--167.8] | 220.4 [213.8--221.3] | 251.5 [223.9--264.1] | 17.70 [17.54--18.66] | 0.0% |
| 4 | llama-server, `cache_prompt: true` | 23.8 [22.1--24.3] | 131.0 [129.9--132.3] | 85.5 [84.9--87.3] | 197.2 [196.5--198.4] | 37.96 [37.85--38.07] | 84.2% |

Cache-hit ratio is cache-hit tokens divided by cache-hit plus newly evaluated prompt tokens. For RadixForge it comes from the delta of `/metrics` counters. For llama-server it is aggregated from terminal streaming `timings.cache_n` and `timings.prompt_n` fields. The two counters are comparable as an indication of prompt work avoided, but not guaranteed to have identical internal definitions.

## Performance fixes

The earlier cache-on concurrency-1 result for RadixForge was 16.8 ms TTFT, 126.2 ms E2E p50, and 7.79 req/s; the post-first-token portion was much slower than llama-server despite similar TTFT. A three-second `sample` capture during a long streaming request showed `llama_sampler_sample` occupying the worker thread, including `ggml_backend_sched_synchronize` / Metal `waitUntilCompleted`, followed by substantial `llama_sampler_top_p_apply` work. The server created a temperature sampler, top-p sampler, and distribution sampler even for `temperature: 0` requests.

RadixForge now creates llama.cpp's greedy sampler directly when `temperature <= 0`, avoiding the full-vocabulary temperature/top-p/distribution sampling path while preserving greedy output. On the same c=1 cache-on workload, E2E p50 fell from 126.2 ms to 67.8 ms and throughput rose from 7.79 to 14.48 req/s; TTFT fell from 16.8 ms to 12.3 ms. The byte output for a representative 24-token Qwen completion was identical before and after the change. Cache-on c=4 throughput rose from 17.80 to 30.29 req/s.

A remaining gap is visible for cache-on E2E: 67.8 ms versus llama-server's 46.0 ms at c=1, and 130.7 ms versus 85.5 ms at c=4. Sampling still synchronizes Metal work once per generated token, as it does in llama-server. The remaining difference also includes separate worker scheduling, per-token callback/SSE handling, and different batching/slot implementations; this run does not attribute that gap to a single component.

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

llama-server remains faster for the cache-enabled end-to-end path in this test, and it has a lower c=1 cache-on E2E median. Its c=4 cache-on median is also lower but has a much wider p95 TTFT. This does not establish a universal winner: llama-server's `cache_prompt` operates through server slots and request scheduling, whereas RadixForge keeps a radix tree that can share a matching prefix across physical sequences, including retained decoded output. The synthetic workload favors one common prefix and does not exercise divergent trees, multi-turn generated-token reuse, eviction pressure, larger models, or mixed arrival patterns.

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
