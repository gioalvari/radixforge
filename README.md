# RadixForge

[![CI](https://github.com/gioalvari/radixforge/actions/workflows/ci.yml/badge.svg)](https://github.com/gioalvari/radixforge/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**Radix Tree KV Cache Orchestrator for multi-agent LLM inference on Apple Silicon.**

RadixForge is a C++ LLM inference server that solves the core multi-agent problem: when 10 AI agents share the same 2000-token system prompt, why compute it 10 times? RadixForge computes it once and shares it in memory — **zero-copy**.

Built directly on top of the native [llama.cpp](https://github.com/ggerganov/llama.cpp) APIs, with no custom GPU kernels. The math is llama.cpp's; the orchestration logic is RadixForge's.

---

## How It Works

```
[Agent A Request]  [Agent B Request]  [Agent C Request]
        |                  |                  |
        └─────────┬─────────┘                  |
                  ▼                            |
      [Radix Tree Orchestrator]  ◄─────────────┘
      (finds longest common prefix)
                  |
      ┌───────────┴──────────────┐
      ▼                          ▼
 [Shared prefix]           [Agent-specific delta]
 (already in KV cache)     (needs to be computed)
      |                          |
      └──────────┬───────────────┘
                 ▼
   [llama_memory_seq_cp]   ← zero-copy into VRAM
                 |
   [llama_decode on Metal] ← delta tokens only (~76% savings)
                 |
   [SSE Streaming Response]
```

### The Radix Tree

The Radix Tree lives entirely on CPU and is the central data structure. Each node contains:
- A token vector (the node's "key")
- A `ref_count` (how many agents are currently using this node)
- A set of `llama_seq_id` values (which physical KV cache slots hold the computed data)

When two requests share a prefix, the tree **splits** the node at the divergence point and allocates a branch for each agent. VRAM is allocated in separate physical `seq_id` slots, but the prefix content is copied at zero cost via `llama_memory_seq_cp`.

### The KV Mapper

The KV Mapper translates Radix Tree logical operations into llama.cpp API calls:

- `prepare_sequence(tokens)` → finds the prefix in the tree, copies the cache, returns the delta tokens
- `release_sequence(seq_id)` → keeps the `seq_id` alive in cache for future reuse (does not free it)
- `gc_if_needed()` → LRU eviction when physical slots run out

### The HTTP Server

Single-threaded HTTP server (cpp-httplib, no external dependencies) with a single `InferenceWorker` thread. This is intentional: Apple Silicon has no hardware GPU multi-tenancy — a single thread managing a queue is more efficient than N threads contending for Metal.

---

## Requirements

- **macOS 13+** (Ventura or later)
- **Apple Silicon** (M1 / M2 / M3 / M4 — any variant)
- **Xcode Command Line Tools**: `xcode-select --install`
- **CMake 3.21+**: `brew install cmake`
- **Git**
- A model file in **GGUF** format

---

## Build

```bash
# 1. Clone the project (llama.cpp is a pinned submodule)
git clone --recursive https://github.com/gioalvari/radixforge
cd radixforge

# 2. One-command build (fetches the llama.cpp submodule, configures Metal, compiles)
chmod +x scripts/setup.sh
./scripts/setup.sh
```

The binary is produced at `build/radixforge` (~120KB stripped). The first run is slower because Metal compiles the kernels JIT (~30 seconds); subsequent runs use the cache.

### Manual build

```bash
git submodule update --init --recursive

cmake -B build -S . \
  -DCMAKE_BUILD_TYPE=Release \
  -DRADIXFORGE_METAL=ON \
  -DCMAKE_OSX_ARCHITECTURES=arm64

cmake --build build --config Release -j$(sysctl -n hw.ncpu)
```

### Tests

```bash
ctest --test-dir build --output-on-failure
```

---

## Usage

### Starting the server

```bash
./build/radixforge \
  -m /path/to/model.gguf \
  --ctx-size 8192 \
  --max-seq 16 \
  --port 8400
```

### CLI options

| Flag | Default | Description |
|------|---------|-------------|
| `-m, --model` | *(required)* | Path to the `.gguf` model file |
| `-c, --ctx-size` | `32768` | Total KV cache size (tokens) |
| `-b, --batch-size` | `2048` | Maximum decode batch size |
| `-ngl, --n-gpu-layers` | `99` | Layers offloaded to Metal (99 = all) |
| `--host` | `127.0.0.1` | Bind address |
| `--port` | `8400` | HTTP port |
| `--max-seq` | `32` | Maximum concurrent sequences |

### Configuration guide

| Mac unified memory | Recommended `--ctx-size` | `--max-seq` |
|-------------------|--------------------------|-------------|
| 16 GB | 8192 | 8 |
| 32 GB | 16384 | 16 |
| 64 GB | 32768 | 32 |
| 128 GB | 65536 | 64 |

---

## API

RadixForge exposes an OpenAI-compatible API. Any client that uses the OpenAI API works without changes by pointing to `http://localhost:8400`.

### `GET /health`

```bash
curl http://localhost:8400/health
```

```json
{"status": "ok", "active_sequences": 3, "pending_requests": 0}
```

### `POST /v1/chat/completions`

```bash
curl http://localhost:8400/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "What is 3+3?"}
    ],
    "max_tokens": 100,
    "temperature": 0.7,
    "stream": false
  }'
```

```json
{
  "choices": [{
    "message": {"role": "assistant", "content": "3 + 3 = 6"},
    "finish_reason": "stop"
  }]
}
```

**Streaming response (SSE):**

```bash
curl http://localhost:8400/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Write a short poem."}],
    "max_tokens": 200,
    "stream": true
  }'
```

Each SSE chunk is a JSON object with `delta.content`. The last chunk carries `finish_reason: "stop"` and the `data: [DONE]` sentinel.

### Using with VS Code (Continue, Cody, Jan)

Any extension that supports custom OpenAI endpoints can point to `http://127.0.0.1:8400` with an arbitrary API key.

---

## Multi-agent example: prefix sharing in action

This is the primary use case. Two or more agents share the same system prompt:

```bash
SHARED_PROMPT="You are a math expert. Answer concisely."

# Agent A — first request (cache MISS: full prefill)
curl http://localhost:8400/v1/chat/completions -H "Content-Type: application/json" -d "{
  \"messages\": [
    {\"role\": \"system\", \"content\": \"$SHARED_PROMPT\"},
    {\"role\": \"user\",   \"content\": \"What is 3+3?\"}
  ],
  \"max_tokens\": 20, \"stream\": false
}"

# Agent B — second request with same system prompt (cache HIT: skips prefix)
curl http://localhost:8400/v1/chat/completions -H "Content-Type: application/json" -d "{
  \"messages\": [
    {\"role\": \"system\", \"content\": \"$SHARED_PROMPT\"},
    {\"role\": \"user\",   \"content\": \"What is 7+7?\"}
  ],
  \"max_tokens\": 20, \"stream\": false
}"
```

**Expected log output:**

```
[radixforge] Prefill: 37 tokens, cached: 0,  delta: 37  ← Agent A: full compute
[radixforge] KV cache hit: copied 28 tokens from seq 1 → seq 2
[radixforge] Prefill: 37 tokens, cached: 28, delta: 9   ← Agent B: 9 delta tokens (76% savings)
```

---

## Project structure

```
radixforge/
├── CMakeLists.txt              # Build system — llama.cpp via add_subdirectory
├── scripts/
│   └── setup.sh               # One-command build script
├── include/radixforge/
│   ├── config.h               # Runtime configuration struct
│   ├── llama_bridge.h         # Phase 1: RAII wrapper for llama.cpp
│   ├── radix_tree.h           # Phase 2: Radix Tree (core data structure)
│   ├── kv_mapper.h            # Phase 3: Virtual→physical KV cache mapper
│   ├── server.h               # Phase 4: HTTP server + InferenceWorker
│   └── json_minimal.h         # Recursive-descent JSON parser (no external deps)
├── src/
│   ├── main.cpp               # Entry point + CLI parsing
│   ├── llama_bridge.cpp       # llama.cpp integration (decode, sample, memory_seq_cp)
│   ├── radix_tree.cpp         # Tree logic (insert, split, evict, LRU)
│   ├── kv_mapper.cpp          # GC, eviction, cache synchronization
│   └── server.cpp             # HTTP (httplib), SSE streaming, generation loop
├── vendor/
│   └── llama.cpp/             # llama.cpp (git submodule, pinned commit)
└── models/                    # Model files directory (git-ignored)
```

---

## Architecture notes

### Why a single worker thread?

Apple Silicon has no hardware GPU multi-tenancy (unlike NVIDIA with MIG). If you launch two `llama_decode` calls on separate threads, Metal serializes them in the command queue anyway. A single worker with explicit batching is the optimal strategy: less context-switch overhead, more tokens per Metal command.

### Why does `seq_cp` always copy the full buffer?

The `llama_memory_seq_cp` API in llama.cpp (v0.15+) only supports partial copies within the same "stream" (contiguous physical sequence). For cross-stream copies — which occur when src and dst belong to different channels — it requires `p0=0, p1=-1` (full copy). RadixForge works around this by copying everything and then calling `llama_memory_seq_rm(dst, matched_tokens, -1)` to discard the excess tokens.

### LRU Eviction

When the physical `seq_id` pool is exhausted, `KVMapper::evict_one()` finds the Radix Tree leaf node with the oldest `last_access_tick` and calls `llama_memory_seq_rm` to free the VRAM. The node is removed from the tree. The next request matching that prefix recomputes it from scratch.

### `find_covering_seq_id()`

After a node split, the `seq_ids` **stay on the prefix (parent) node** — the suffix (child) node starts with an empty `seq_ids` set. `find_covering_seq_id()` checks the parent node before descending into children: this lets new requests that match in the suffix still find the physical cache of the prefix, without recomputing already-cached tokens.

---

## Known limitations

- **macOS / Apple Silicon only** — the Metal backend is hardcoded. Linux/NVIDIA is technically feasible (remove Metal frameworks from CMake) but untested.
- **Single model** — the server loads one GGUF model at startup. Serving different models in parallel requires separate instances on different ports.
- **Fixed chat template** — uses ChatML (`<|im_start|>role\ncontent<|im_end|>`). Models with different templates (Llama-3, Mistral) may produce degraded quality.
- **No tool use / function calling** — text completions only.

---

## Recommended models

Any GGUF model works. Some tested options:

| Model | Size | Quality | RAM needed |
|-------|------|---------|------------|
| `Qwen2.5-0.5B-Instruct-Q4_K_M` | 469 MB | Test/dev | 2 GB |
| `Qwen2.5-7B-Instruct-Q4_K_M` | 4.7 GB | ★★★★☆ | 8 GB |
| `Llama-3.2-3B-Instruct-Q4_K_M` | 2.0 GB | ★★★☆☆ | 4 GB |
| `Mistral-7B-Instruct-v0.3-Q4_K_M` | 4.4 GB | ★★★★☆ | 8 GB |

Download from [Hugging Face](https://huggingface.co/models?sort=trending&search=gguf&pipeline_tag=text-generation).

---

## License

MIT — see `LICENSE`.
