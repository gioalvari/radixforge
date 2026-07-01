#pragma once

#include <cstdint>
#include <memory>
#include <set>
#include <shared_mutex>
#include <unordered_map>
#include <vector>

#include "llama.h"

namespace radixforge {

struct RadixNode {
    std::vector<llama_token> tokens;
    int32_t ref_count = 0;
    std::set<llama_seq_id> seq_ids;  // physical KV cache sequences sharing this node
    uint64_t last_access_tick = 0;
    // Length of the canonical KV sequence at the time it was established.
    // Used to trim the canonical seq back to this boundary after a full-cache-hit
    // decode_single (which appends one extra position that must not accumulate).
    int32_t canonical_len = 0;

    bool has_physical_cache() const { return !seq_ids.empty(); }
    llama_seq_id any_seq_id() const { return *seq_ids.begin(); }

    // Find any seq_id covering this node's prefix.
    // After a split, seq_ids stay on the prefix node (parent), so we check the
    // parent first before searching children. Trim to match.matched_tokens in
    // prepare_sequence makes it safe to use a parent's longer KV sequence.
    llama_seq_id find_covering_seq_id(int32_t* out_canonical_len = nullptr) const {
        if (!seq_ids.empty()) {
            if (out_canonical_len) *out_canonical_len = canonical_len;
            return *seq_ids.begin();
        }
        // Parent may hold seq_ids if this node is a suffix created by split_node.
        if (parent && !parent->seq_ids.empty()) {
            if (out_canonical_len) *out_canonical_len = parent->canonical_len;
            return *parent->seq_ids.begin();
        }
        for (const auto& [key, child] : children) {
            int32_t clen = 0;
            llama_seq_id sid = child->find_covering_seq_id(&clen);
            if (sid != -1) {
                if (out_canonical_len) *out_canonical_len = clen;
                return sid;
            }
        }
        return -1;
    }

    std::unordered_map<llama_token, std::unique_ptr<RadixNode>> children;
    RadixNode* parent = nullptr;
};

struct PrefixMatch {
    RadixNode* node;              // leaf node for this request (new or existing)
    RadixNode* cache_node;        // node holding physical KV cache for matched_tokens
    llama_seq_id cache_seq_id = -1; // seq_id to copy from (set during partial-match split)
    int32_t matched_tokens;       // total tokens matched along the path
    std::vector<llama_token> remaining;  // tokens not yet in cache
};

class RadixTree {
public:
    RadixTree();

    // Insert a full prompt token sequence. Returns the prefix match info
    // and prepares the tree for the new branch.
    PrefixMatch insert(const std::vector<llama_token>& prompt_tokens);

    // Mark a node (and its ancestors) as no longer referenced by an agent.
    void release(RadixNode* node);

    // Find LRU leaf nodes for eviction. Returns nodes sorted by last_access_tick.
    std::vector<RadixNode*> find_eviction_candidates(int32_t count);

    // Remove a leaf node from the tree entirely.
    void evict(RadixNode* node);

    // Get current number of active sequences (sum of all seq_ids across nodes).
    int32_t active_sequence_count() const;

    RadixNode* root() { return root_.get(); }

private:
    std::unique_ptr<RadixNode> root_;
    uint64_t tick_ = 0;
    mutable std::shared_mutex mutex_;  // shared for reads, exclusive for writes

    PrefixMatch find_prefix_locked(const std::vector<llama_token>& tokens);
    void split_node(RadixNode* node, size_t split_pos);
};

} // namespace radixforge
