#include "radixforge/kv_mapper.h"
#include "radixforge/logger.h"

#include <algorithm>
#include <stdexcept>

namespace radixforge {

KVMapper::KVMapper(KvOps& kv_ops, RadixTree& tree, const Config& config)
    : kv_ops_(kv_ops), tree_(tree), config_(config) {
    // Initialize free seq_id pool (0 is reserved for scratch)
    for (int32_t i = 1; i < config.max_sequences; i++) {
        free_seq_ids_.push_back(i);
    }
}

KVMapper::PrepareResult KVMapper::prepare_sequence(const std::vector<llama_token>& prompt_tokens) {

    metrics_.total_requests.fetch_add(1, std::memory_order_relaxed);
    metrics_.prompt_tokens.fetch_add(prompt_tokens.size(), std::memory_order_relaxed);

    evict_kv_pressure();

    // Step 1: Find longest prefix match in radix tree
    PrefixMatch match = tree_.insert(prompt_tokens);

    llama_seq_id new_seq = -1;
    int32_t cached_pos = 0;
    try {
        // Hold the mapper lock while choosing and copying a ready source.
        // Every nested tree operation follows the mapper -> tree lock order
        // documented in the header.
        std::lock_guard<std::mutex> lock(mutex_);
        // Allocate before changing the tree's physical registration. If the pool
        // is exhausted, release the reference acquired by insert() below.
        new_seq = allocate_seq_id_locked();

        // Look up the source after allocation/eviction, so an id returned here
        // cannot have been evicted by allocate_seq_id().
        int32_t source_len = 0;
        llama_seq_id src_seq = match.matched_tokens > 0
                                   ? tree_.find_covering_seq_id(match.cache_node, &source_len)
                                   : -1;
        cached_pos = src_seq != -1 ? std::min(match.matched_tokens, source_len) : 0;

        // A non-empty full hit still decodes its final prompt token to obtain
        // logits. Copy only the preceding KV cells so that token is decoded at
        // its original position instead of being duplicated at position n.
        if (!prompt_tokens.empty() && cached_pos == static_cast<int32_t>(prompt_tokens.size())) {
            --cached_pos;
        }

        if (cached_pos > 0) {
            // kv_unified puts every seq_id in one stream. llama.cpp therefore
            // tags just this prefix's existing cells with new_seq (zero-copy).
            kv_ops_.seq_cp(src_seq, new_seq, 0, cached_pos);
            metrics_.hits.fetch_add(1, std::memory_order_relaxed);
            metrics_.cache_hit_tokens.fetch_add(cached_pos, std::memory_order_relaxed);
            RF_DEBUG("kv_mapper", "KV cache hit: copied %d tokens from seq %d -> seq %d",
                     cached_pos, src_seq, new_seq);
        } else if (match.matched_tokens > 0) {
            // Node exists in tree but no seq_id found anywhere in subtree — recompute
            metrics_.misses.fetch_add(1, std::memory_order_relaxed);
            RF_DEBUG("kv_mapper", "Prefix match (%d tokens) but no physical cache, will recompute",
                     match.matched_tokens);
            RF_DEBUG("kv_mapper", "No ready source; pending sources are never copied");
        } else {
            metrics_.misses.fetch_add(1, std::memory_order_relaxed);
        }

        tree_.register_seq_id(match.node, new_seq, static_cast<int32_t>(prompt_tokens.size()));
        seq_cell_counts_[new_seq] = cached_pos;
        retained_seq_ids_.erase(new_seq);
    } catch (...) {
        if (new_seq != -1) {
            kv_ops_.seq_rm(new_seq, 0, -1);
            std::lock_guard<std::mutex> lock(mutex_);
            free_seq_ids_.push_back(new_seq);
            seq_cell_counts_.erase(new_seq);
        }
        tree_.release(match.node);
        throw;
    }

    PrepareResult result;
    result.seq_id = new_seq;
    result.cached_pos = cached_pos;
    result.delta.assign(prompt_tokens.begin() + cached_pos, prompt_tokens.end());
    result.leaf_node = match.node;
    return result;
}

void KVMapper::mark_sequence_ready(llama_seq_id seq_id, RadixNode* node) {
    std::lock_guard<std::mutex> lock(mutex_);
    tree_.mark_seq_id_ready(node, seq_id);
    if (node)
        seq_cell_counts_[seq_id] = node->canonical_len;
}

void KVMapper::set_sequence_cell_count(llama_seq_id seq_id, int32_t cell_count) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (seq_cell_counts_.count(seq_id) != 0) {
        seq_cell_counts_[seq_id] = std::max(0, cell_count);
    }
}

RadixNode* KVMapper::extend_sequence(llama_seq_id seq_id, RadixNode* node,
                                     const std::vector<llama_token>& tokens) {
    std::lock_guard<std::mutex> lock(mutex_);
    RadixNode* extended =
        tree_.extend_seq_id(node, seq_id, tokens, static_cast<int32_t>(tokens.size()));
    if (seq_cell_counts_.count(seq_id) != 0) {
        seq_cell_counts_[seq_id] = static_cast<int32_t>(tokens.size());
    }
    return extended;
}

PrefixAvailability KVMapper::inspect_prefix(const std::vector<llama_token>& prompt_tokens) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return tree_.inspect_prefix(prompt_tokens);
}

void KVMapper::release_sequence(llama_seq_id seq_id, RadixNode* node) {
    std::lock_guard<std::mutex> lock(mutex_);

    bool should_free = tree_.release_seq_id(node, seq_id);
    if (should_free) {
        kv_ops_.seq_rm(seq_id, 0, -1);
        free_seq_ids_.push_back(seq_id);
        seq_cell_counts_.erase(seq_id);
        retained_seq_ids_.erase(seq_id);
    } else {
        int32_t canonical_len = 0;
        if (tree_.find_covering_seq_id(node, &canonical_len) == seq_id) {
            kv_ops_.seq_rm(seq_id, canonical_len, -1);
            seq_cell_counts_[seq_id] = canonical_len;
            retained_seq_ids_.insert(seq_id);
        }
    }
}

void KVMapper::gc_if_needed() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!free_seq_ids_.empty())
        return;
    RF_INFO("kv_mapper", "GC triggered: evicting LRU sequences");
    evict_one_locked();
}

bool KVMapper::evict_idle_sequence() {
    std::lock_guard<std::mutex> lock(mutex_);
    return evict_one_locked();
}

void KVMapper::evict_kv_pressure() {
    std::lock_guard<std::mutex> lock(mutex_);
    evict_kv_pressure_locked();
}

int64_t KVMapper::kv_cells_used() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return kv_cells_used_locked();
}

int32_t KVMapper::active_sequence_count() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return static_cast<int32_t>(seq_cell_counts_.size() - retained_seq_ids_.size());
}

int32_t KVMapper::retained_sequence_count() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return static_cast<int32_t>(retained_seq_ids_.size());
}

llama_seq_id KVMapper::allocate_seq_id_locked() {
    if (free_seq_ids_.empty()) {
        evict_one_locked();
    }
    if (free_seq_ids_.empty()) {
        throw SequenceCapacityExhausted("No free seq_ids available after eviction");
    }
    // Warn when pool is near saturation (below gc_watermark remaining)
    int32_t remaining = static_cast<int32_t>(free_seq_ids_.size());
    if (remaining <= (config_.max_sequences - config_.gc_watermark)) {
        RF_WARN("kv_mapper", "Seq pool near saturation: %d/%d free slots remaining", remaining,
                config_.max_sequences);
    }
    llama_seq_id id = free_seq_ids_.front();
    free_seq_ids_.pop_front();
    return id;
}

bool KVMapper::evict_one_locked() {
    while (true) {
        auto candidates = tree_.find_eviction_candidates(1);
        if (candidates.empty()) {
            RF_WARN("kv_mapper", "No eviction candidates available");
            return false;
        }

        std::vector<llama_seq_id> evicted = tree_.evict(candidates[0]);
        if (evicted.empty())
            continue;

        for (llama_seq_id sid : evicted) {
            kv_ops_.seq_rm(sid, 0, -1);
            free_seq_ids_.push_back(sid);
            seq_cell_counts_.erase(sid);
            retained_seq_ids_.erase(sid);
            RF_DEBUG("kv_mapper", "Evicted seq %d (LRU)", sid);
        }
        metrics_.evictions.fetch_add(evicted.size(), std::memory_order_relaxed);
        return true;
    }
}

bool KVMapper::evict_kv_pressure_locked() {
    const int64_t high_watermark =
        static_cast<int64_t>(static_cast<double>(config_.n_ctx) * config_.kv_high_watermark);
    bool evicted = false;
    while (kv_cells_used_locked() > high_watermark) {
        if (!evict_one_locked())
            break;
        evicted = true;
    }
    return evicted;
}

int64_t KVMapper::kv_cells_used_locked() const {
    int64_t total = 0;
    for (const auto& [seq_id, cells] : seq_cell_counts_)
        total += cells;
    return total;
}

void KVMapper::defrag() {
    // Proactive GC every N requests keeps the pool healthy.
    gc_if_needed();
}

} // namespace radixforge
