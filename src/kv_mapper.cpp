#include "radixforge/kv_mapper.h"

#include <cstdio>
#include <stdexcept>

namespace radixforge {

KVMapper::KVMapper(LlamaBridge& bridge, RadixTree& tree, const Config& config)
    : bridge_(bridge), tree_(tree), config_(config) {
    // Initialize free seq_id pool (0 is reserved for scratch)
    for (int32_t i = 1; i < config.max_sequences; i++) {
        free_seq_ids_.push_back(i);
    }
}

KVMapper::PrepareResult KVMapper::prepare_sequence(
    const std::vector<llama_token>& prompt_tokens) {

    // Step 1: Find longest prefix match in radix tree
    PrefixMatch match = tree_.insert(prompt_tokens);

    // Step 2: Allocate a physical seq_id (may trigger GC)
    gc_if_needed();
    llama_seq_id new_seq = allocate_seq_id();

    // Step 3: Copy cached KV prefix if available.
    // Resolve source seq_id and canonical_len:
    //   a) cache_seq_id != -1 (partial node match after split): captured before split
    //   b) cache_node has seq_ids: use the canonical seq directly
    //   c) cache_node has no seq_ids but a descendant does: use its seq_id
    llama_seq_id src_seq = -1;

    if (match.cache_seq_id != -1) {
        src_seq = match.cache_seq_id;
    } else if (match.matched_tokens > 0) {
        src_seq = match.cache_node->find_covering_seq_id(nullptr);
    }

    if (src_seq != -1) {
        // Copy the full canonical seq, then trim to exactly matched_tokens.
        // This removes any extra position that may have been appended by a
        // previous full-cache-hit decode (which decodes at position cached_pos,
        // one beyond the last stored token).
        bridge_.memory_seq_cp(src_seq, new_seq, 0, -1);
        bridge_.memory_seq_rm(new_seq, match.matched_tokens, -1);

        fprintf(stderr, "[radixforge] KV cache hit: copied %d tokens from seq %d → seq %d\n",
                match.matched_tokens, src_seq, new_seq);
    } else if (match.matched_tokens > 0) {
        // Node exists in tree but no seq_id found anywhere in subtree — must recompute
        fprintf(stderr, "[radixforge] Prefix match (%d tokens) but no physical cache, "
                "will recompute\n", match.matched_tokens);
        match.remaining.insert(match.remaining.begin(),
                               prompt_tokens.begin(),
                               prompt_tokens.begin() + match.matched_tokens);
        match.matched_tokens = 0;
    }

    // Register physical seq_id on the deepest matched node
    {
        std::lock_guard<std::mutex> lock(mutex_);
        match.node->seq_ids.insert(new_seq);
        if (match.node->canonical_len == 0) {
            // First seq_id for this node — record the stable canonical length
            match.node->canonical_len = match.matched_tokens;
        }
    }

    PrepareResult result;
    result.seq_id = new_seq;
    result.cached_pos = match.matched_tokens;
    result.delta = std::move(match.remaining);
    result.leaf_node = match.node;
    return result;
}

void KVMapper::release_sequence(llama_seq_id seq_id, RadixNode* node) {
    // After generation completes, keep ONE canonical seq_id per node so future
    // requests can copy from it. Use a mutex to prevent the double-free race
    // condition where two concurrent releases both see size==2 and both free.
    std::lock_guard<std::mutex> lock(mutex_);

    bool should_free = true;
    if (node) {
        auto it = node->seq_ids.find(seq_id);
        if (it != node->seq_ids.end()) {
            if (node->seq_ids.size() <= 1) {
                // This is the ONLY seq_id — keep it as the canonical cached entry.
                should_free = false;
            } else {
                // Node already has another cached seq_id → release this one.
                node->seq_ids.erase(it);
            }
        }
    }
    if (should_free) {
        bridge_.memory_seq_rm(seq_id, 0, -1);
        free_seq_ids_.push_back(seq_id);
    }
}

void KVMapper::gc_if_needed() {
    std::lock_guard<std::mutex> lock(mutex_);
    if ((int32_t)free_seq_ids_.size() > 0) return;

    // Need to evict — find LRU candidates
    fprintf(stderr, "[radixforge] GC triggered: evicting LRU sequences\n");
    evict_one();
}

llama_seq_id KVMapper::allocate_seq_id() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (free_seq_ids_.empty()) {
        evict_one();
    }
    if (free_seq_ids_.empty()) {
        throw std::runtime_error("No free seq_ids available after eviction");
    }
    llama_seq_id id = free_seq_ids_.front();
    free_seq_ids_.pop_front();
    return id;
}

void KVMapper::evict_one() {
    // Find the LRU leaf node and destroy its physical cache
    auto candidates = tree_.find_eviction_candidates(1);
    if (candidates.empty()) {
        fprintf(stderr, "[radixforge] WARNING: No eviction candidates available\n");
        return;
    }

    RadixNode* victim = candidates[0];

    // Remove all physical sequences from this node
    for (llama_seq_id sid : victim->seq_ids) {
        bridge_.memory_seq_rm(sid, 0, -1);
        free_seq_ids_.push_back(sid);
        fprintf(stderr, "[radixforge] Evicted seq %d (LRU)\n", sid);
    }
    victim->seq_ids.clear();

    tree_.evict(victim);
}

void KVMapper::defrag() {
    // Defrag is now automatic in llama.cpp via defrag_thold parameter.
    // This is a no-op kept for interface compatibility.
}

} // namespace radixforge
