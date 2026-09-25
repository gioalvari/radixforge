#pragma once

#include "radixforge/config.h"
#include "radixforge/llama_bridge.h"
#include "radixforge/radix_tree.h"

#include <atomic>
#include <cstdint>
#include <deque>
#include <mutex>
#include <vector>

namespace radixforge {

// Snapshot of cache performance counters. All fields are atomics so they
// can be read from the HTTP thread while the worker updates them.
struct CacheMetrics {
    std::atomic<uint64_t> hits{0};          // requests where KV cache was copied
    std::atomic<uint64_t> misses{0};        // requests where full prefill was needed
    std::atomic<uint64_t> evictions{0};     // physical seq_ids freed by LRU eviction
    std::atomic<uint64_t> total_requests{0};

    CacheMetrics() = default;
    CacheMetrics(const CacheMetrics&) = delete;
    CacheMetrics& operator=(const CacheMetrics&) = delete;
};

// Manages the mapping between logical Radix Tree nodes and physical
// llama_seq_id slots in the KV cache.
class KVMapper {
public:
    KVMapper(LlamaBridge& bridge, RadixTree& tree, const Config& config);

    // Prepare a sequence for generation:
    // 1. Find prefix match in the radix tree
    // 2. Allocate a physical seq_id
    // 3. Copy cached KV entries via llama_memory_seq_cp
    // 4. Return the remaining tokens that need decoding
    struct PrepareResult {
        llama_seq_id seq_id;
        int32_t cached_pos;              // position up to which cache is valid
        std::vector<llama_token> delta;  // tokens still needing decode
        RadixNode* leaf_node;
    };

    PrepareResult prepare_sequence(const std::vector<llama_token>& prompt_tokens);

    // Mark a prepared sequence eligible as a prefix-copy source after its
    // prefill llama_decode call has succeeded.
    void mark_sequence_ready(llama_seq_id seq_id, RadixNode* node);

    // Inspect ready and pending cache coverage without allocating a seq_id.
    PrefixAvailability inspect_prefix(const std::vector<llama_token>& prompt_tokens) const;

    // Release a physical seq_id back to the pool after generation is done.
    void release_sequence(llama_seq_id seq_id, RadixNode* node);

    // Forcefully evict a specific seq_id (admin use only). Returns false if not found.
    bool force_evict_seq(llama_seq_id seq_id);

    // Run proactive garbage collection (evict idle LRU sequences).
    void gc_if_needed();

    // Defragment the underlying KV cache (also runs gc_if_needed).
    void defrag();

    // Read-only access to cache performance counters.
    const CacheMetrics& metrics() const { return metrics_; }

private:
    LlamaBridge& bridge_;
    RadixTree& tree_;
    Config config_;

    std::deque<llama_seq_id> free_seq_ids_;
    // Lock order when both locks are needed is KVMapper::mutex_, then
    // RadixTree's shared_mutex. RadixTree never acquires the mapper mutex.
    mutable std::mutex mutex_;
    CacheMetrics metrics_;

    llama_seq_id allocate_seq_id_locked();
    void evict_one_locked();
};

} // namespace radixforge
