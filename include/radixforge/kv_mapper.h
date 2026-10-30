#pragma once

#include "radixforge/config.h"
#include "radixforge/llama_bridge.h"
#include "radixforge/radix_tree.h"

#include <atomic>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace radixforge {

class SequenceCapacityExhausted : public std::runtime_error {
  public:
    using std::runtime_error::runtime_error;
};

// Snapshot of cache performance counters. All fields are atomics so they
// can be read from the HTTP thread while the worker updates them.
struct CacheMetrics {
    std::atomic<uint64_t> hits{0};      // requests where KV cache was copied
    std::atomic<uint64_t> misses{0};    // requests where full prefill was needed
    std::atomic<uint64_t> evictions{0}; // physical seq_ids freed by LRU eviction
    std::atomic<uint64_t> total_requests{0};
    std::atomic<uint64_t> cache_hit_tokens{0};
    std::atomic<uint64_t> prompt_tokens{0};

    CacheMetrics() = default;
    CacheMetrics(const CacheMetrics&) = delete;
    CacheMetrics& operator=(const CacheMetrics&) = delete;
};

// Minimal KV-cache operations used by KVMapper. Tests can supply a recording
// implementation without constructing a model context.
class KvOps {
  public:
    virtual ~KvOps() = default;
    virtual void seq_cp(llama_seq_id src, llama_seq_id dst, int32_t p0, int32_t p1) = 0;
    virtual void seq_rm(llama_seq_id seq_id, int32_t p0, int32_t p1) = 0;
};

class LlamaKvOps final : public KvOps {
  public:
    explicit LlamaKvOps(LlamaBridge& bridge) : bridge_(bridge) {}

    void seq_cp(llama_seq_id src, llama_seq_id dst, int32_t p0, int32_t p1) override;
    void seq_rm(llama_seq_id seq_id, int32_t p0, int32_t p1) override;

  private:
    LlamaBridge& bridge_;
};

// Manages the mapping between logical Radix Tree nodes and physical
// llama_seq_id slots in the KV cache.
class KVMapper {
  public:
    KVMapper(LlamaBridge& bridge, RadixTree& tree, const Config& config);
    KVMapper(KvOps& kv_ops, RadixTree& tree, const Config& config);

    // Prepare a sequence for generation:
    // 1. Find prefix match in the radix tree
    // 2. Allocate a physical seq_id
    // 3. Copy cached KV entries via llama_memory_seq_cp
    // 4. Return the remaining tokens that need decoding
    struct PrepareResult {
        llama_seq_id seq_id;
        int32_t cached_pos; // position up to which cache is valid
        // Tokens still needing decode. delta is non-empty for non-empty prompts.
        std::vector<llama_token> delta;
        RadixNode* leaf_node;
    };

    PrepareResult prepare_sequence(const std::vector<llama_token>& prompt_tokens);

    // Mark a prepared sequence eligible as a prefix-copy source after its
    // prefill llama_decode call has succeeded.
    void mark_sequence_ready(llama_seq_id seq_id, RadixNode* node);

    // Record the number of decoded KV cells currently held by a sequence.
    void set_sequence_cell_count(llama_seq_id seq_id, int32_t cell_count);

    // Associate a ready sequence with its complete valid KV token prefix.
    // The returned node owns seq_id; callers retain their original node ref.
    RadixNode* extend_sequence(llama_seq_id seq_id, RadixNode* node,
                               const std::vector<llama_token>& tokens);

    // Inspect ready and pending cache coverage without allocating a seq_id.
    PrefixAvailability inspect_prefix(const std::vector<llama_token>& prompt_tokens) const;

    // Release a physical seq_id back to the pool after generation is done.
    void release_sequence(llama_seq_id seq_id, RadixNode* node);

    // Run proactive garbage collection (evict idle LRU sequences).
    void gc_if_needed();

    // Defragment the underlying KV cache (also runs gc_if_needed).
    void defrag();

    // Evict an idle cached sequence when available. Referenced nodes (active or
    // pending requests) are excluded by RadixTree's ref_count check.
    bool evict_idle_sequence();

    // Evict idle retained sequences until tracked KV-cell use is below the
    // configured high-watermark. Called automatically before preparation.
    void evict_kv_pressure();

    int64_t kv_cells_used() const;
    int32_t active_sequence_count() const;
    int32_t retained_sequence_count() const;

    // Read-only access to cache performance counters.
    const CacheMetrics& metrics() const {
        return metrics_;
    }

  private:
    std::unique_ptr<KvOps> owned_kv_ops_;
    KvOps& kv_ops_;
    RadixTree& tree_;
    Config config_;

    std::deque<llama_seq_id> free_seq_ids_;
    std::unordered_map<llama_seq_id, int32_t> seq_cell_counts_;
    std::unordered_set<llama_seq_id> retained_seq_ids_;
    // Lock order when both locks are needed is KVMapper::mutex_, then
    // RadixTree's shared_mutex. RadixTree never acquires the mapper mutex.
    mutable std::mutex mutex_;
    CacheMetrics metrics_;

    llama_seq_id allocate_seq_id_locked();
    bool evict_one_locked();
    bool evict_kv_pressure_locked();
    int64_t kv_cells_used_locked() const;
};

} // namespace radixforge
