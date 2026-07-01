#pragma once

#include "radixforge/config.h"
#include "radixforge/llama_bridge.h"
#include "radixforge/radix_tree.h"

#include <deque>
#include <mutex>
#include <vector>

namespace radixforge {

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

    // Release a physical seq_id back to the pool after generation is done.
    void release_sequence(llama_seq_id seq_id, RadixNode* node);

    // Run garbage collection if needed (evict LRU sequences).
    void gc_if_needed();

    // Defragment the underlying KV cache.
    void defrag();

private:
    LlamaBridge& bridge_;
    RadixTree& tree_;
    Config config_;

    std::deque<llama_seq_id> free_seq_ids_;
    std::mutex mutex_;

    llama_seq_id allocate_seq_id();
    void evict_one();
};

} // namespace radixforge
