# Position-independent KV chunk reuse (EPIC-lite) spike

**Status:** both required 100-item runs complete.

## Setup

- Repository: `feature/chunk-kv`, llama.cpp submodule `f728adab`.
- Model run: `Qwen2.5-0.5B-Instruct-Q4_K_M.gguf`, Metal enabled, `-ngl 99`.
- Context: 8192, `kv_unified=true`, F16 K/V, four sequence IDs.  Prefix KV was
  computed once and then shared with `llama_memory_seq_cp` at unchanged
  positions.  Thus the reported TTFT excludes prefix work for every variant.
- Dataset: 100 deterministic items, RNG seed `20260925`, sorted by ID before
  sampling and after sampling.  The official CMU URL timed out on this machine,
  so `scripts/prepare_hotpot.py` used the specified HF
  `hotpotqa/hotpot_qa` / `distractor` validation parquet fallback.  It uses only
  Python stdlib; it invokes the installed `hf` and `duckdb` executables for the
  fallback conversion.  Every paragraph is whitespace-normalized and capped at
  120 words.
- Prompt is the requested fixed Qwen ChatML layout.  `llama_tokenize` is called
  with `parse_special=true`; only the prefix receives `add_special=true`.
  The full reference is decoded from the **concatenation of these already
  tokenized pieces**, rather than re-tokenizing the full string.
- Measurements were made while running on battery:
  `28%; discharging; 1:02 remaining`.  Treat timings as indicative.

Run command:

```sh
./build/tools/chunk_spike \
  -m /Users/a470718/local-llm-bench/models/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf \
  --data data/hotpot_dev_100.jsonl --n 100 \
  --variants full,naive,prefixed,prefixed+r8,prefixed+r16,prefixed+r32,prefixed+r64,naive+r16,prefixed+rALL \
  --out /tmp/spike-0.5b.jsonl
```

The 7B partial download finished at `4,683,074,240` bytes (within the required
approximate `4.68e9` check), its first four bytes were `GGUF`, and its curl
process had exited before it was renamed to its final `.gguf` path.

## 0.5B results (n=100)

The first-answer distribution uses full-vocabulary float64 softmax KL
`KL(full || variant)`.  EM and F1 use SQuAD normalization. `tokens computed`
  counts request computation.  Prefix work is excluded from the measured TTFT
  for every row, but included in the `full` token-computation accounting;
  variants count only suffix and explicitly recomputed chunk tokens because the
  prefix and chunk tails are cached.

| variant | EM | F1 | agree-with-full | top-1 | mean KL | median TTFT ms | tokens computed |
|---|---:|---:|---:|---:|---:|---:|---:|
| full | 0.1200 | 0.2654 | 1.0000 | 1.0000 | 0.0000 | 255.2790 | 1269.3500 |
| naive | 0.0400 | 0.1219 | 0.0600 | 0.3000 | 2.8576 | 17.1890 | 29.6500 |
| prefixed | 0.1400 | 0.2208 | 0.2700 | 0.4100 | 1.3167 | 17.7329 | 29.6500 |
| prefixed+r8 | 0.0900 | 0.1840 | 0.3000 | 0.4900 | 0.9623 | 122.8868 | 101.6500 |
| prefixed+r16 | 0.0800 | 0.1768 | 0.3300 | 0.4900 | 0.8575 | 136.6715 | 173.6500 |
| prefixed+r32 | 0.1100 | 0.2138 | 0.4000 | 0.6000 | 0.7660 | 137.4814 | 317.3300 |
| prefixed+r64 | 0.1200 | 0.2366 | 0.5300 | 0.6600 | 0.4265 | 172.2218 | 593.4900 |
| naive+r16 | 0.0900 | 0.1773 | 0.2500 | 0.4500 | 1.1609 | 138.9115 | 173.6500 |
| prefixed+rALL | 0.1200 | 0.2654 | 1.0000 | 1.0000 | 0.0000 | 255.4168 | 1269.3500 |

One-off cache construction computes both naïve and prefixed canonical blobs
for all chunks.  It is recorded per JSONL row as `chunk_precompute_ms` and is
excluded from TTFT because it is amortized cache population.  Blob storage and
bytes/token are likewise recorded per row.  The full raw output is
`/tmp/spike-0.5b.jsonl` (900 records).

## API findings and mechanics

The pinned API supports the required calls:

- `llama_state_seq_get_data_ext` / `llama_state_seq_set_data_ext` serialize and
  restore a sequence state.  The restored state gets fresh cache cells.
- With unified KV, `llama_memory_seq_cp` does **not** copy K/V: it adds the
  destination sequence ID to existing cells.  Consequently,
  `llama_memory_seq_add` on an aliased sequence would change the physical cell
  position for every alias.  Canonical blobs are never shifted in place.
- The safe attachment path restores each canonical blob into sequence 2,
  removes its prefix (for prefixed blobs) and recomputed head, shifts only the
  temporary tail, synchronizes to charge lazy K-shift to attach time, aliases
  the tail onto target sequence 1, then removes sequence 2.
- llama.cpp enforces consecutive positions when a sequence already owns KV
  cells.  Target construction is therefore strictly left-to-right: shared
  prefix, fresh recomputed head, shifted attached tail, and suffix.
- `llama_memory_can_shift` is checked before processing.

The explicit independence probe restores a canonical prefixed blob, obtains a
probe-token logit vector, performs a shifted temporary attach from another
restore, restores the canonical blob again, and compares vectors.  All 100
items reported `independence_max_abs = 0`.  `prefixed+rALL` is the end-to-end
sanity check: its aggregate KL is `0.0000`, top-1 and greedy answer agreement
are `1.0000`, and its TTFT is indistinguishable from full.  This establishes
that differences in the other rows come from the intended KV composition, not
piecewise tokenization or cache corruption.

## Assessment and caveats

Prefix-aware cached chunks are substantially less divergent than naïve chunks
at the same near-zero recomputation cost (KL 1.3167 vs 2.8576).  On this small
model, `prefixed+r64` is the best measured quality/cost point: it cuts request
token computation by about 53% and median TTFT by about 33%, while raising
top-1 agreement to 0.66 and reducing mean KL to 0.4265.  It is **not** close
enough to declare position-independent reuse production-equivalent: greedy
agreement remains 0.53 and KL remains material.  Larger recompute windows or
architectural changes are required for fidelity.

## 7B results (n=100)

The 7B run used the same data, configuration, variants, and output procedure.
It finished before the 40-minute fallback threshold, so it retained `n=100`.
The battery state when results were collected had changed to AC power:
`47%; charging; 1:42 remaining`.

| variant | EM | F1 | agree-with-full | top-1 | mean KL | median TTFT ms | tokens computed |
|---|---:|---:|---:|---:|---:|---:|---:|
| full | 0.4800 | 0.6334 | 1.0000 | 1.0000 | 0.0000 | 3493.8867 | 1269.3500 |
| naive | 0.1200 | 0.2416 | 0.2000 | 0.3700 | 4.6491 | 112.2202 | 29.6500 |
| prefixed | 0.2200 | 0.3786 | 0.3600 | 0.4700 | 3.0798 | 111.4415 | 29.6500 |
| prefixed+r8 | 0.2400 | 0.3774 | 0.3900 | 0.5000 | 2.7277 | 1409.6271 | 101.6500 |
| prefixed+r16 | 0.2400 | 0.3863 | 0.4200 | 0.5000 | 2.9941 | 1027.8325 | 173.6500 |
| prefixed+r32 | 0.2900 | 0.4319 | 0.5000 | 0.6000 | 2.7102 | 1012.9906 | 317.3300 |
| prefixed+r64 | 0.4000 | 0.5453 | 0.7200 | 0.7900 | 0.8789 | 1637.4610 | 593.4900 |
| naive+r16 | 0.1500 | 0.2826 | 0.3100 | 0.4400 | 3.2739 | 1029.9313 | 173.6500 |
| prefixed+rALL | 0.4800 | 0.6334 | 1.0000 | 1.0000 | 0.0000 | 3532.6230 | 1269.3500 |

The raw output is `/tmp/spike-7b.jsonl` (900 records).  The rALL and
independence checks passed on all items exactly as for 0.5B.

Main risks are model/architecture dependence, generation-level drift that a
first-token metric understates, cache import/export overhead and host-device
transfer behavior, lazy K-shift cost, context pressure from temporary restored
blobs, and API semantics changing as llama.cpp evolves.  Results must be
confirmed with the 7B model on stable power before selecting a production N.

## Relevance-guided recompute (7B, n=52, AC power)

Hypothesis: in RAG and agent memory the retriever already knows which chunks
matter, so recomputing only those chunks fully (at their final positions,
attending to the prefix and all earlier composed chunks) should recover most of
the quality at a fraction of the cost. New variants:

- `prefixed+gold`: recompute the HotpotQA supporting paragraphs (oracle selector);
- `prefixed+topK`: recompute the top-K paragraphs by BM25 (question vs paragraph,
  k1 = 1.2, b = 0.75, IDF over the item's 10 paragraphs);
- `prefixed+top2+r16`: top-2 fully plus the first 16 tokens of every other chunk.

Recomputed tokens are now buffered and decoded in as few `llama_decode` calls as
possible (a flush is only needed before attaching cached cells) and the suffix is
merged into the last call; `decode calls` below is the mean per request. K-shift
is no longer synchronized per attached chunk.

BM25 top-2 contains both gold paragraphs in 41% of items, top-3 in 56%.

The run was stopped by a 2-hour shell limit after 52 complete items (all 9
variants each); the remaining 48 were not run. Numbers below are the 52 items,
measured on AC power. Absolute times are higher than in the 100-item run above
(full prefill 5.0 s vs 3.5 s median); compare rows with each other.

| variant | EM | F1 | agree-with-full | top-1 | mean KL | median KL | median TTFT ms | tokens computed | decode calls |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| full | 0.52 | 0.67 | 1.00 | 1.00 | 0.000 | 0.000 | 5025 | 1299 | 1.0 |
| prefixed | 0.21 | 0.36 | 0.35 | 0.46 | 2.940 | 0.741 | 188 | 28 | 1.0 |
| prefixed+r16 | 0.21 | 0.33 | 0.37 | 0.50 | 2.823 | 0.427 | 1654 | 172 | 10.0 |
| prefixed+r64 | 0.40 | 0.55 | 0.69 | 0.79 | 0.877 | 0.009 | 2745 | 599 | 9.0 |
| prefixed+gold | 0.37 | 0.53 | 0.60 | 0.73 | 1.065 | 0.044 | 1236 | 242 | 2.6 |
| prefixed+top2 | 0.35 | 0.50 | 0.58 | 0.65 | 1.880 | 0.184 | 1239 | 245 | 2.7 |
| prefixed+top3 | 0.35 | 0.48 | 0.63 | 0.71 | 0.987 | 0.075 | 1657 | 344 | 3.2 |
| prefixed+top2+r16 | 0.37 | 0.50 | 0.67 | 0.79 | 0.841 | 0.021 | 2120 | 360 | 8.3 |
| prefixed+rALL | 0.52 | 0.67 | 1.00 | 1.00 | 0.000 | 0.000 | 5174 | 1275 | 1.0 |

Sanity: `prefixed+rALL` KL = 0 and canonical-blob independence = 0 on all items.

### Assessment

- **Relevance-guided recompute is the best trade-off, but not good enough.**
  BM25 top-2 cuts TTFT about 4× and keeps 67% of the exact-match score
  (0.35 vs 0.52).
- **The selector is not the bottleneck.** Even the oracle (`gold`) only reaches
  0.37. Chunks that were never computed in the context they are inserted into
  degrade the answer, and fully recomputing the relevant chunks does not repair
  that.
- **There is no large fixed overhead left to remove.** With fewer decode calls,
  7B prefill still costs ~3.5–4 ms per recomputed token; savings are
  proportional to the tokens skipped.
- **Conclusion.** Training-free reuse built on llama.cpp's public API (import,
  shift, recompute whole tokens) loses about a third of the answer quality on
  multi-document QA with a 7B model. Recovering it would need CacheBlend-style
  per-layer selective recompute, which requires changes to llama.cpp's graph and
  Metal kernels. The work is paused; the tool stays in `tools/chunk_spike.cpp`
  for reproduction:

```bash
python3 scripts/prepare_hotpot.py
./build/tools/chunk_spike -m <model.gguf> --data data/hotpot_dev_100.jsonl --n 100 \
  --variants full,prefixed,prefixed+r64,prefixed+gold,prefixed+top2,prefixed+rALL \
  --out results.jsonl
```
