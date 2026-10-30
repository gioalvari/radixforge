# Shared-prefix server comparison

**Run date:** 2026-10-30

**RadixForge commit:** `d355896a8688b2e5886411b932fb22cee87c735c`

**Vendored llama.cpp commit:** `f728adab683387b2bdb085b3854a0f3487a13d0a`

**Model:** `Qwen2.5-0.5B-Instruct-Q4_K_M.gguf`

**Hardware:** Apple M4 Pro; 51,539,607,552 bytes of memory; macOS Metal backend.

This is a single-machine, small-model comparison. It is a reproducible engineering measurement, not a general performance claim.

## Results

Each cell is the median of three fresh-process repetitions. The bracket is the min--max range across repetitions. Every repetition measured 24 streaming `/completion` requests after one excluded warm-up request.

| Concurrency | Server and cache mode | TTFT p50 (ms) | TTFT p95 (ms) | E2E p50 (ms) | E2E p95 (ms) | Throughput (req/s) | Cache-hit token ratio |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | RadixForge, `--no-prefix-cache` | 45.1 [44.6--48.2] | 56.4 [54.1--125.8] | 142.6 [128.2--163.1] | 258.1 [245.8--339.9] | 5.93 [5.86--6.28] | 0.0% |
| 1 | RadixForge, cache enabled | 16.8 [15.5--18.1] | 28.6 [24.2--50.7] | 126.2 [108.7--131.5] | 169.8 [132.7--236.0] | 7.79 [6.71--8.94] | 96.3% |
| 1 | llama-server, `cache_prompt: false` | 43.7 [43.3--43.9] | 47.9 [44.3--52.1] | 88.7 [84.6--90.5] | 105.1 [94.6--115.2] | 10.93 [10.64--11.58] | 0.0% |
| 1 | llama-server, `cache_prompt: true` | 14.0 [13.9--16.6] | 19.4 [18.1--31.8] | 57.4 [56.1--70.3] | 94.5 [75.6--121.2] | 16.59 [13.22--16.96] | 96.3% |
| 4 | RadixForge, `--no-prefix-cache` | 126.7 [125.5--163.7] | 193.1 [173.8--230.4] | 342.7 [332.8--361.9] | 588.4 [367.0--756.6] | 9.75 [9.46--11.58] | 0.0% |
| 4 | RadixForge, cache enabled | 33.2 [33.1--33.2] | 41.9 [37.0--61.1] | 223.3 [208.9--254.6] | 240.2 [224.7--287.5] | 17.80 [16.33--18.56] | 96.3% |
| 4 | llama-server, `cache_prompt: false` | 159.7 [159.2--176.5] | 171.3 [169.9--198.3] | 251.0 [241.5--284.6] | 322.6 [299.7--442.2] | 15.35 [12.68--15.37] | 0.0% |
| 4 | llama-server, `cache_prompt: true` | 28.5 [27.2--28.9] | 142.9 [138.1--152.6] | 107.6 [97.9--109.1] | 242.1 [220.6--244.8] | 30.63 [29.72--32.15] | 84.2% |

Cache-hit ratio is cache-hit tokens divided by cache-hit plus newly evaluated prompt tokens. For RadixForge it comes from the delta of `/metrics` counters. For llama-server it is aggregated from terminal streaming `timings.cache_n` and `timings.prompt_n` fields. The two counters are comparable as an indication of prompt work avoided, but not guaranteed to have identical internal definitions.

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

On this workload, both servers substantially reduce first-token latency when prompt caching is enabled. RadixForge moves from 45.1 ms to 16.8 ms median TTFT at concurrency 1 and from 126.7 ms to 33.2 ms at concurrency 4; its metrics show 96.3% prompt-token reuse. This validates the `--no-prefix-cache` A/B control on identical prompts.

llama-server was faster in this particular small-model setup in both cache modes and at both tested concurrency levels. Its cache-enabled median throughput was 16.59 req/s at concurrency 1 and 30.63 req/s at concurrency 4, compared with RadixForge's 7.79 and 17.80 req/s. Its concurrency-4 p95 TTFT is much wider despite a lower median, so tail behavior deserves separate study rather than a single winner label.

This test does not establish that one cache design is universally superior. llama-server's `cache_prompt` operates through its server slots and request scheduling; RadixForge keeps a radix tree of reusable prefixes and can share a matching prefix across physical sequences, including retained decoded output. The synthetic workload favors a single common prefix and does not exercise divergent trees, multi-turn generated-token reuse, eviction pressure, larger models, or mixed arrival patterns.

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
