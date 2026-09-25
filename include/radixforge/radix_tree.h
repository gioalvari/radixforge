#pragma once

#include <cstdint>
#include <memory>
#include <set>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "llama.h"

namespace radixforge {

struct RadixNode {
    std::vector<llama_token> tokens;
    int32_t ref_count = 0;
    // All registered physical KV sequences. A sequence is registered as pending
    // before its prefill decode and moves to ready only after that decode succeeds.
    std::set<llama_seq_id> seq_ids;
    std::set<llama_seq_id> ready_seq_ids;
    uint64_t last_access_tick = 0;
    // Length of the canonical KV sequence at the time it was established.
    // Used to trim the canonical seq back to this boundary after a full-cache-hit
    // decode_single (which appends one extra position that must not accumulate).
    int32_t canonical_len = 0;

    bool has_physical_cache() const { return !ready_seq_ids.empty(); }
    llama_seq_id any_seq_id() const { return *ready_seq_ids.begin(); }

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

// Non-mutating prefix/cache inspection used to avoid copying from a sequence
// whose prefill is still pending in the same worker batch.
struct PrefixAvailability {
    int32_t matched_tokens = 0;
    int32_t ready_tokens = 0;
    int32_t pending_tokens = 0;
};

class RadixTree {
public:
    RadixTree();

    // Insert a full prompt token sequence. Returns the prefix match info
    // and prepares the tree for the new branch.
    PrefixMatch insert(const std::vector<llama_token>& prompt_tokens);

    // Inspect the longest logical prefix and the best ready/pending physical
    // coverage without acquiring a request reference or modifying the tree.
    PrefixAvailability inspect_prefix(const std::vector<llama_token>& prompt_tokens) const;

    // Mark a node (and its ancestors) as no longer referenced by an agent.
    void release(RadixNode* node);

    // Find LRU nodes that own idle physical KV sequences. Returns nodes sorted by
    // last_access_tick. Internal nodes are valid candidates after a radix split.
    std::vector<RadixNode*> find_eviction_candidates(int32_t count);

    // Remove physical sequences from an idle node. Leaves are removed from the
    // tree; internal nodes retain their logical prefix structure. Returns the
    // invalidated sequence ids, which must be released from llama's KV cache.
    std::vector<llama_seq_id> evict(RadixNode* node);

    // Register a physical sequence as pending, then mark it ready only after
    // llama_decode has populated its prompt KV. Returns true from
    // release_seq_id when the caller should free the sequence. A sole ready
    // sequence is retained as the canonical cache entry; pending sequences
    // are always removed and freed.
    void register_seq_id(RadixNode* node, llama_seq_id seq_id,
                          int32_t canonical_len);
    void mark_seq_id_ready(RadixNode* node, llama_seq_id seq_id);
    bool release_seq_id(RadixNode* node, llama_seq_id seq_id);
    bool erase_seq_id(llama_seq_id seq_id);

    // Find a physical sequence covering node's prefix under the tree lock.
    // Evicted ids are removed before this method can return them.
    llama_seq_id find_covering_seq_id(const RadixNode* node,
                                      int32_t* out_canonical_len = nullptr) const;

    // Get current number of active sequences (sum of all seq_ids across nodes).
    int32_t active_sequence_count() const;

    // Serialize the tree structure to a JSON string for diagnostics.
    std::string to_json() const;

    RadixNode* root() { return root_.get(); }

private:
    std::unique_ptr<RadixNode> root_;
    uint64_t tick_ = 0;
    mutable std::shared_mutex mutex_;  // shared for reads, exclusive for writes

    PrefixMatch find_prefix_locked(const std::vector<llama_token>& tokens);
    PrefixAvailability inspect_prefix_locked(const std::vector<llama_token>& tokens) const;
    void split_node(RadixNode* node, size_t split_pos);
    llama_seq_id find_covering_seq_id_locked(
        const RadixNode* node, int32_t* out_canonical_len) const;
    static void consider_availability(const RadixNode* node, int32_t covered,
                                      PrefixAvailability* availability);
};

} // namespace radixforge
