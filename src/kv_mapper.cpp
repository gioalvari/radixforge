#include "radixforge/kv_mapper.h"
#include "radixforge/logger.h"

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

    metrics_.total_requests.fetch_add(1, std::memory_order_relaxed);

    // Step 1: Find longest prefix match in radix tree
    PrefixMatch match = tree_.insert(prompt_tokens);

    // Step 2: Allocate a physical seq_id — handles eviction internally if needed.
    // NOTE: gc_if_needed() is intentionally NOT called here; allocate_seq_id()
    // is the single authoritative eviction point to avoid double eviction.
    llama_seq_id new_seq = allocate_seq_id();

    // Step 3: Copy cached KV prefix if available.
    llama_seq_id src_seq = -1;

    if (match.cache_seq_id != -1) {
        src_seq = match.cache_seq_id;
    } else if (match.matched_tokens > 0) {
        src_seq = match.cache_node->find_covering_seq_id(nullptr);
    }

    if (src_seq != -1) {
        bridge_.memory_seq_cp(src_seq, new_seq, 0, -1);
        bridge_.memory_seq_rm(new_seq, match.matched_tokens, -1);
        metrics_.hits.fetch_add(1, std::memory_order_relaxed);
        RF_DEBUG("kv_mapper", "KV cache hit: copied %d tokens from seq %d -> seq %d",
                 match.matched_tokens, src_seq, new_seq);
    } else if (match.matched_tokens > 0) {
        // Node exists in tree but no seq_id found anywhere in subtree — recompute
        metrics_.misses.fetch_add(1, std::memory_order_relaxed);
        RF_DEBUG("kv_mapper", "Prefix match (%d tokens) but no physical cache, will recompute",
                 match.matched_tokens);
        match.remaining.insert(match.remaining.begin(),
                               prompt_tokens.begin(),
                               prompt_tokens.begin() + match.matched_tokens);
        match.matched_tokens = 0;
    } else {
        metrics_.misses.fetch_add(1, std::memory_order_relaxed);
    }

    // Register physical seq_id on the deepest matched node
    {
        std::lock_guard<std::mutex> lock(mutex_);
        match.node->seq_ids.insert(new_seq);
        if (match.node->canonical_len == 0) {
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
    std::lock_guard<std::mutex> lock(mutex_);

    bool should_free = true;
    if (node) {
        auto it = node->seq_ids.find(seq_id);
        if (it != node->seq_ids.end()) {
            if (node->seq_ids.size() <= 1) {
                // Keep this as the canonical cached entry for future requests.
                should_free = false;
            } else {
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
    if (!free_seq_ids_.empty()) return;
    RF_INFO("kv_mapper", "GC triggered: evicting LRU sequences");
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
    // Warn when pool is near saturation (below gc_watermark remaining)
    int32_t remaining = static_cast<int32_t>(free_seq_ids_.size());
    if (remaining <= (config_.max_sequences - config_.gc_watermark)) {
        RF_WARN("kv_mapper", "Seq pool near saturation: %d/%d free slots remaining",
                remaining, config_.max_sequences);
    }
    llama_seq_id id = free_seq_ids_.front();
    free_seq_ids_.pop_front();
    return id;
}

void KVMapper::evict_one() {
    auto candidates = tree_.find_eviction_candidates(1);
    if (candidates.empty()) {
        RF_WARN("kv_mapper", "No eviction candidates available");
        return;
    }

    RadixNode* victim = candidates[0];
    uint32_t n_evicted = 0;
    for (llama_seq_id sid : victim->seq_ids) {
        bridge_.memory_seq_rm(sid, 0, -1);
        free_seq_ids_.push_back(sid);
        ++n_evicted;
        RF_DEBUG("kv_mapper", "Evicted seq %d (LRU)", sid);
    }
    victim->seq_ids.clear();
    metrics_.evictions.fetch_add(n_evicted, std::memory_order_relaxed);

    tree_.evict(victim);
}

void KVMapper::defrag() {
    // Proactive GC every N requests keeps the pool healthy.
    gc_if_needed();
}

bool KVMapper::force_evict_seq(llama_seq_id target_seq) {
    std::lock_guard<std::mutex> lock(mutex_);

    // Search all nodes for the target seq_id
    std::vector<RadixNode*> stk = {tree_.root()};
    RadixNode* owner = nullptr;
    while (!stk.empty()) {
        RadixNode* n = stk.back(); stk.pop_back();
        if (n->seq_ids.count(target_seq)) { owner = n; break; }
        for (auto& [k, c] : n->children) stk.push_back(c.get());
    }

    // Also check free_seq_ids — seq might already be freed
    for (auto id : free_seq_ids_) {
        if (id == target_seq) return false;  // already free, nothing to do
    }

    bridge_.memory_seq_rm(target_seq, 0, -1);
    free_seq_ids_.push_back(target_seq);
    metrics_.evictions.fetch_add(1, std::memory_order_relaxed);

    if (owner) owner->seq_ids.erase(target_seq);

    RF_INFO("kv_mapper", "Admin: force-evicted seq %d", target_seq);
    return true;
}

} // namespace radixforge
