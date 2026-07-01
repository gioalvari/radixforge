# RadixForge

**Radix Tree KV Cache Orchestrator for multi-agent LLM inference on Apple Silicon.**

RadixForge è un server di inferenza LLM scritto in C++ che risolve il problema fondamentale del multi-agent: quando 10 agenti AI condividono lo stesso system prompt da 2000 token, perché ricalcolarlo 10 volte? RadixForge lo calcola una volta sola e lo condivide in memoria — **zero-copy**.

Costruito direttamente sopra le API native di [llama.cpp](https://github.com/ggerganov/llama.cpp), senza kernel GPU personalizzati. La matematica è di llama.cpp; il cervello di orchestrazione è RadixForge.

---

## Come funziona

```
[Richiesta Agente A]  [Richiesta Agente B]  [Richiesta Agente C]
        |                     |                     |
        └──────────┬──────────┘                     |
                   ▼                                |
       [Radix Tree Orchestrator]  ◄─────────────────┘
       (trova il prefisso comune)
                   |
       ┌───────────┴──────────────┐
       ▼                          ▼
  [Prefisso condiviso]     [Delta specifico]
  (già in KV cache)        (da calcolare)
       |                          |
       └────────┬─────────────────┘
                ▼
    [llama_memory_seq_cp]   ← copia zero-copy in VRAM
                |
    [llama_decode su Metal] ← solo i token delta (risparmio ~76%)
                |
    [SSE Streaming Response]
```

### Il Radix Tree

Il Radix Tree vive interamente su CPU ed è la struttura dati centrale. Ogni nodo contiene:
- Un vettore di token (la "chiave" del nodo)
- Un `ref_count` (quanti agenti stanno usando questo nodo)
- Un insieme di `llama_seq_id` (quali slot fisici nella KV cache contengono i dati calcolati)

Quando due richieste condividono un prefisso, il tree **splitta** il nodo al punto di divergenza e assegna un ramo per ogni agente. La VRAM viene allocata in `seq_id` fisici separati, ma il contenuto del prefisso viene copiato a costo zero tramite `llama_memory_seq_cp`.

### Il KV Mapper

Il KV Mapper traduce le operazioni logiche del Radix Tree in chiamate alle API di llama.cpp:

- `prepare_sequence(tokens)` → trova il prefisso nel tree, copia la cache, restituisce i token delta
- `release_sequence(seq_id)` → mantiene il `seq_id` attivo nella cache per riuso futuro (non lo libera)
- `gc_if_needed()` → LRU eviction quando i canali fisici si esauriscono

### Il Server HTTP

Server HTTP (cpp-httplib, zero dipendenze esterne) con un singolo `InferenceWorker` thread. Questo è intenzionale: Apple Silicon non ha multi-tenancy hardware della GPU — un singolo thread che gestisce una coda è più efficiente di N thread che si contendono Metal.

---

## Requisiti

- **macOS 13+** (Ventura o superiore)
- **Apple Silicon** (M1 / M2 / M3 / M4 — qualsiasi variante)
- **Xcode Command Line Tools**: `xcode-select --install`
- **CMake 3.21+**: `brew install cmake`
- **Git**
- Un file modello in formato **GGUF**

---

## Build

```bash
# 1. Clona il progetto
git clone <url-repo> radixforge
cd radixforge

# 2. Build automatico (clona llama.cpp, configura Metal, compila)
chmod +x scripts/setup.sh
./scripts/setup.sh
```

Il binario viene prodotto in `build/radixforge` (~120KB stripped). La prima esecuzione è più lenta perché Metal compila i kernel JIT (~30 secondi); le esecuzioni successive usano la cache.

### Build manuale

```bash
git clone --depth 1 https://github.com/ggerganov/llama.cpp vendor/llama.cpp

cmake -B build -S . \
  -DCMAKE_BUILD_TYPE=Release \
  -DRADIXFORGE_METAL=ON \
  -DCMAKE_OSX_ARCHITECTURES=arm64

cmake --build build --config Release -j$(sysctl -n hw.ncpu)
```

---

## Utilizzo

### Avvio

```bash
./build/radixforge \
  -m /percorso/al/modello.gguf \
  --ctx-size 8192 \
  --max-seq 16 \
  --port 8400
```

### Opzioni CLI

| Flag | Default | Descrizione |
|------|---------|-------------|
| `-m, --model` | *(obbligatorio)* | Percorso al file `.gguf` |
| `-c, --ctx-size` | `32768` | Dimensione totale della KV cache (token) |
| `-b, --batch-size` | `2048` | Dimensione massima del batch di decode |
| `-ngl, --n-gpu-layers` | `99` | Layer offloaded su Metal (99 = tutto) |
| `--host` | `127.0.0.1` | Indirizzo di ascolto |
| `--port` | `8400` | Porta HTTP |
| `--max-seq` | `32` | Numero massimo di sequenze concorrenti |

### Consigli sulla configurazione

| Memoria unificata Mac | `--ctx-size` consigliato | `--max-seq` |
|----------------------|--------------------------|-------------|
| 16 GB | 8192 | 8 |
| 32 GB | 16384 | 16 |
| 64 GB | 32768 | 32 |
| 128 GB | 65536 | 64 |

---

## API

RadixForge espone un'API compatibile con OpenAI. Qualsiasi client che usa l'API OpenAI funziona senza modifiche puntando a `http://localhost:8400`.

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
      {"role": "system", "content": "Sei un assistente utile."},
      {"role": "user", "content": "Quanto fa 3+3?"}
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

**Risposta in streaming (SSE):**

```bash
curl http://localhost:8400/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Scrivi una poesia breve."}],
    "max_tokens": 200,
    "stream": true
  }'
```

Ogni chunk SSE è un JSON con `delta.content`. L'ultimo chunk ha `finish_reason: "stop"` e il segnale `data: [DONE]`.

### Uso con VS Code (Copilot-compatible client)

Qualsiasi estensione che supporta endpoint OpenAI custom (es. **Continue**, **Cody**, **Jan**) può puntare a `http://127.0.0.1:8400` con una chiave API arbitraria.

---

## Esempio multi-agent: prefix sharing in azione

Questo è il caso d'uso principale. Due o più agenti condividono lo stesso system prompt:

```bash
SHARED_PROMPT="Sei un esperto di matematica. Rispondi in modo conciso."

# Agente A — prima richiesta (cache MISS: calcola tutto)
curl http://localhost:8400/v1/chat/completions -H "Content-Type: application/json" -d "{
  \"messages\": [
    {\"role\": \"system\", \"content\": \"$SHARED_PROMPT\"},
    {\"role\": \"user\",   \"content\": \"Quanto fa 3+3?\"}
  ],
  \"max_tokens\": 20, \"stream\": false
}"

# Agente B — seconda richiesta con stesso system prompt (cache HIT: salta il prefisso)
curl http://localhost:8400/v1/chat/completions -H "Content-Type: application/json" -d "{
  \"messages\": [
    {\"role\": \"system\", \"content\": \"$SHARED_PROMPT\"},
    {\"role\": \"user\",   \"content\": \"Quanto fa 7+7?\"}
  ],
  \"max_tokens\": 20, \"stream\": false
}"
```

**Log atteso:**

```
[radixforge] Prefill: 37 tokens, cached: 0,  delta: 37  ← Agente A: calcola tutto
[radixforge] KV cache hit: copied 28 tokens from seq 1 → seq 2
[radixforge] Prefill: 37 tokens, cached: 28, delta: 9   ← Agente B: solo 9 token delta (76% risparmio)
```

---

## Struttura del progetto

```
radixforge/
├── CMakeLists.txt              # Build system — llama.cpp via add_subdirectory
├── scripts/
│   └── setup.sh               # Build one-command
├── include/radixforge/
│   ├── config.h               # Struct di configurazione runtime
│   ├── llama_bridge.h         # Fase 1: RAII wrapper per llama.cpp
│   ├── radix_tree.h           # Fase 2: Radix Tree (struttura dati core)
│   ├── kv_mapper.h            # Fase 3: Mapper virtuale→fisico della KV cache
│   ├── server.h               # Fase 4: HTTP server + InferenceWorker
│   └── json_minimal.h         # Parser JSON recursive-descent (no deps esterni)
├── src/
│   ├── main.cpp               # Entry point + parsing CLI
│   ├── llama_bridge.cpp       # Integrazione llama.cpp (decode, sample, memory_seq_cp)
│   ├── radix_tree.cpp         # Logica dell'albero (insert, split, evict, LRU)
│   ├── kv_mapper.cpp          # GC, eviction, sincronizzazione cache
│   └── server.cpp             # HTTP (httplib), SSE streaming, loop generazione
├── vendor/
│   └── llama.cpp/             # llama.cpp clonato (non submodule)
└── models/                    # Cartella modelli (ignorata da git)
```

---

## Architettura interna: note tecniche

### Perché un singolo worker thread?

Apple Silicon non ha multi-tenancy hardware della GPU (a differenza di NVIDIA con MIG). Se lanci due `llama_decode` su thread diversi, Metal li serializza comunque in coda. Un singolo worker con batching esplicito è la strategia ottimale: meno overhead di context switch, più token per singolo comando Metal.

### Perché `seq_cp` copia sempre il buffer intero?

L'API `llama_memory_seq_cp` di llama.cpp (v0.15+) supporta copie parziali solo dentro lo stesso "stream" (sequenza fisica contigua). Per copie cross-stream — che si verificano quando src e dst appartengono a canali diversi — richiede `p0=0, p1=-1` (full copy). RadixForge aggira questo copiando tutto e poi chiamando `llama_memory_seq_rm(dst, matched_tokens, -1)` per rimuovere i token in eccesso.

### LRU Eviction

Quando il pool di `seq_id` fisici si esaurisce, `KVMapper::evict_one()` trova il nodo foglia del Radix Tree con il `last_access_tick` più vecchio e chiama `llama_memory_seq_rm` per liberare la VRAM. Il nodo viene rimosso dall'albero. La prossima richiesta che corrispondeva a quel prefisso lo ricalcola da zero.

### `find_covering_seq_id()`

Dopo uno split del nodo, i `seq_ids` **restano sul nodo prefisso (padre)** — il nodo suffisso (figlio) parte con `seq_ids` vuoto. `find_covering_seq_id()` controlla prima il nodo padre prima di scendere nei figli: questo permette a nuove richieste che fanno match nel suffisso di trovare comunque la cache fisica del prefisso, senza ricalcolare i token già presenti.

---

## Limitazioni note

- **Solo macOS / Apple Silicon** — il backend Metal è hardcoded. Linux/NVIDIA è tecnicamente supportabile (basta rimuovere i framework Metal dal CMake) ma non testato.
- **Single model** — il server carica un unico modello GGUF all'avvio. Per servire modelli diversi in parallelo sono necessarie istanze separate su porte diverse.
- **Chat template fisso** — usa ChatML (`<|im_start|>role\ncontent<|im_end|>`). Modelli con template diversi (Llama-3, Mistral) potrebbero avere qualità degradata.
- **Nessun tool use / function calling** — solo completions testuali.

---

## Modelli consigliati

Qualsiasi modello GGUF funziona. Alcuni testati:

| Modello | Dimensione | Qualità | RAM necessaria |
|---------|-----------|---------|----------------|
| `Qwen2.5-0.5B-Instruct-Q4_K_M` | 469 MB | Test/dev | 2 GB |
| `Qwen2.5-7B-Instruct-Q4_K_M` | 4.7 GB | ★★★★☆ | 8 GB |
| `Llama-3.2-3B-Instruct-Q4_K_M` | 2.0 GB | ★★★☆☆ | 4 GB |
| `Mistral-7B-Instruct-v0.3-Q4_K_M` | 4.4 GB | ★★★★☆ | 8 GB |

Download da [Hugging Face](https://huggingface.co/models?sort=trending&search=gguf&pipeline_tag=text-generation).

---

## Licenza

MIT — vedi `LICENSE`.
