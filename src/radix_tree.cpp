#include "radixforge/radix_tree.h"

#include <algorithm>
#include <string>

namespace radixforge {

RadixTree::RadixTree() {
    root_ = std::make_unique<RadixNode>();
    root_->ref_count = 1;  // root never gets evicted
}

PrefixMatch RadixTree::insert(const std::vector<llama_token>& prompt_tokens) {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    ++tick_;

    PrefixMatch match = find_prefix_locked(prompt_tokens);

    // The node found by find_prefix is the cache_node (it may have physical KV data).
    match.cache_node = match.node;

    // If there are unmatched tokens, create a new leaf node to hold them.
    // This populates the tree so future requests can share the prefix.
    if (!match.remaining.empty()) {
        auto new_leaf = std::make_unique<RadixNode>();
        new_leaf->tokens = match.remaining;
        new_leaf->parent = match.node;
        new_leaf->last_access_tick = tick_;
        llama_token key = match.remaining[0];
        match.node->children[key] = std::move(new_leaf);
        match.node = match.node->children[key].get();
    }

    match.node->last_access_tick = tick_;
    match.node->ref_count++;

    return match;
}

PrefixAvailability RadixTree::inspect_prefix(
    const std::vector<llama_token>& prompt_tokens) const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    return inspect_prefix_locked(prompt_tokens);
}

void RadixTree::release(RadixNode* node) {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    if (node && node->ref_count > 0) {
        node->ref_count--;
    }
}

PrefixMatch RadixTree::find_prefix_locked(const std::vector<llama_token>& tokens) {
    RadixNode* current = root_.get();
    int32_t matched_total = 0;
    size_t token_idx = 0;
    while (token_idx < tokens.size()) {
        llama_token next_token = tokens[token_idx];

        auto it = current->children.find(next_token);
        if (it == current->children.end()) {
            // No child matches — this is where we diverge
            break;
        }

        RadixNode* child = it->second.get();

        // Match tokens within this child node
        size_t match_len = 0;
        while (match_len < child->tokens.size() &&
               token_idx + match_len < tokens.size() &&
               child->tokens[match_len] == tokens[token_idx + match_len]) {
            match_len++;
        }

        if (match_len < child->tokens.size()) {
            // Partial match within node — split needed.
            split_node(child, match_len);
            // After split, child now contains only the matched prefix
            matched_total += (int32_t)match_len;
            token_idx += match_len;
            current = child;
            break;
        }

        // Full match of this node
        matched_total += (int32_t)child->tokens.size();
        token_idx += child->tokens.size();
        current = child;
    }

    PrefixMatch result;
    result.node = current;
    result.cache_node = current;
    result.matched_tokens = matched_total;
    result.remaining.assign(tokens.begin() + token_idx, tokens.end());
    return result;
}

void RadixTree::consider_availability(const RadixNode* node, int32_t covered,
                                      PrefixAvailability* availability) {
    if (!node || covered <= 0) return;
    const int32_t cache_coverage = std::min(covered, node->canonical_len);
    if (cache_coverage <= 0) return;
    if (!node->ready_seq_ids.empty()) {
        availability->ready_tokens = std::max(availability->ready_tokens, cache_coverage);
    }
    if (node->seq_ids.size() > node->ready_seq_ids.size()) {
        availability->pending_tokens = std::max(availability->pending_tokens, cache_coverage);
    }
}

PrefixAvailability RadixTree::inspect_prefix_locked(
    const std::vector<llama_token>& tokens) const {
    PrefixAvailability availability;
    const RadixNode* current = root_.get();
    size_t token_idx = 0;
    int32_t matched = 0;

    while (token_idx < tokens.size()) {
        auto it = current->children.find(tokens[token_idx]);
        if (it == current->children.end()) break;

        const RadixNode* child = it->second.get();
        size_t match_len = 0;
        while (match_len < child->tokens.size() && token_idx + match_len < tokens.size() &&
               child->tokens[match_len] == tokens[token_idx + match_len]) {
            ++match_len;
        }
        if (match_len == 0) break;

        matched += static_cast<int32_t>(match_len);
        consider_availability(child, matched, &availability);
        if (match_len < child->tokens.size()) break;

        token_idx += match_len;
        current = child;
    }
    availability.matched_tokens = matched;
    return availability;
}

void RadixTree::split_node(RadixNode* node, size_t split_pos) {
    // Create a new child that holds the suffix (the original path beyond split_pos).
    auto suffix_node = std::make_unique<RadixNode>();
    suffix_node->tokens.assign(node->tokens.begin() + split_pos, node->tokens.end());

    // Bug fix (was: MOVE seq_ids to suffix, copy ref_count → both corrupted).
    // seq_ids STAY on the prefix (node): active requests have leaf_node=node and
    // release_sequence looks for their seq_id in node->seq_ids. Moving them to
    // suffix breaks that lookup, causing spurious physical KV destruction.
    // The suffix starts with empty seq_ids; it will accumulate its own once a
    // request decodes through the full suffix path.
    // ref_count STAYS on prefix (active requests point to node, not suffix).
    // Suffix starts at 0 — no request has it as leaf_node yet.
    suffix_node->seq_ids.clear();
    suffix_node->ready_seq_ids.clear();
    suffix_node->canonical_len = 0;
    suffix_node->ref_count = 0;
    suffix_node->last_access_tick = node->last_access_tick;
    suffix_node->parent = node;

    // Move all children of original node to the suffix node
    suffix_node->children = std::move(node->children);
    for (auto& [key, child] : suffix_node->children) {
        child->parent = suffix_node.get();
    }

    // Truncate the original node to prefix only.
    // seq_ids, canonical_len, and ref_count remain on the prefix node.
    node->tokens.resize(split_pos);
    node->children.clear();

    // The suffix becomes the single child of the (now prefix-only) node
    llama_token suffix_key = suffix_node->tokens[0];
    node->children[suffix_key] = std::move(suffix_node);
}

std::vector<RadixNode*> RadixTree::find_eviction_candidates(int32_t count) {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    std::vector<RadixNode*> candidates;

    // A split leaves physical seq_ids on its prefix, which is usually an
    // internal node. Select every idle, non-root owner of physical KV data.
    std::vector<RadixNode*> stack = {root_.get()};
    while (!stack.empty()) {
        RadixNode* node = stack.back();
        stack.pop_back();

        if (node != root_.get() && node->ref_count == 0 &&
            !node->seq_ids.empty()) {
            candidates.push_back(node);
        }
        for (auto& [key, child] : node->children) {
            stack.push_back(child.get());
        }
    }

    // Sort by LRU (oldest access first)
    std::sort(candidates.begin(), candidates.end(), [](RadixNode* a, RadixNode* b) {
        return a->last_access_tick < b->last_access_tick;
    });

    if ((int32_t)candidates.size() > count) {
        candidates.resize(count);
    }
    return candidates;
}

std::vector<llama_seq_id> RadixTree::evict(RadixNode* node) {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    if (!node || !node->parent || node->ref_count != 0 ||
        node->seq_ids.empty()) {
        return {};
    }

    std::vector<llama_seq_id> evicted(node->seq_ids.begin(), node->seq_ids.end());
    node->seq_ids.clear();
    node->ready_seq_ids.clear();
    node->canonical_len = 0;

    // Internal prefix nodes must remain in the tree. Future requests can still
    // match them and safely recompute because they have no physical cache.
    if (!node->children.empty()) return evicted;

    RadixNode* parent = node->parent;
    llama_token key = node->tokens[0];
    parent->children.erase(key);
    return evicted;
}

void RadixTree::register_seq_id(RadixNode* node, llama_seq_id seq_id,
                                int32_t canonical_len) {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    node->seq_ids.insert(seq_id);
    if (node->canonical_len == 0) node->canonical_len = canonical_len;
}

void RadixTree::mark_seq_id_ready(RadixNode* node, llama_seq_id seq_id) {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    if (node && node->seq_ids.count(seq_id) != 0) {
        node->ready_seq_ids.insert(seq_id);
    }
}

bool RadixTree::release_seq_id(RadixNode* node, llama_seq_id seq_id) {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    if (!node) return true;

    auto it = node->seq_ids.find(seq_id);
    if (it == node->seq_ids.end()) return true;
    const bool is_ready = node->ready_seq_ids.count(seq_id) != 0;
    if (is_ready && node->seq_ids.size() <= 1) return false;

    node->seq_ids.erase(it);
    node->ready_seq_ids.erase(seq_id);
    if (node->seq_ids.empty()) node->canonical_len = 0;
    return true;
}

bool RadixTree::erase_seq_id(llama_seq_id seq_id) {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    std::vector<RadixNode*> stack = {root_.get()};
    while (!stack.empty()) {
        RadixNode* node = stack.back();
        stack.pop_back();

        auto it = node->seq_ids.find(seq_id);
        if (it != node->seq_ids.end()) {
            node->seq_ids.erase(it);
            node->ready_seq_ids.erase(seq_id);
            if (node->seq_ids.empty()) node->canonical_len = 0;
            return true;
        }
        for (auto& [key, child] : node->children) {
            stack.push_back(child.get());
        }
    }
    return false;
}

llama_seq_id RadixTree::find_covering_seq_id(
    const RadixNode* node, int32_t* out_canonical_len) const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    return find_covering_seq_id_locked(node, out_canonical_len);
}

llama_seq_id RadixTree::find_covering_seq_id_locked(
    const RadixNode* node, int32_t* out_canonical_len) const {
    if (!node) return -1;
    if (!node->ready_seq_ids.empty()) {
        if (out_canonical_len) *out_canonical_len = node->canonical_len;
        return *node->ready_seq_ids.begin();
    }
    if (node->parent && !node->parent->ready_seq_ids.empty()) {
        if (out_canonical_len) *out_canonical_len = node->parent->canonical_len;
        return *node->parent->ready_seq_ids.begin();
    }
    for (const auto& [key, child] : node->children) {
        int32_t canonical_len = 0;
        llama_seq_id seq_id =
            find_covering_seq_id_locked(child.get(), &canonical_len);
        if (seq_id != -1) {
            if (out_canonical_len) *out_canonical_len = canonical_len;
            return seq_id;
        }
    }
    return -1;
}

int32_t RadixTree::active_sequence_count() const {
    std::shared_lock<std::shared_mutex> lock(mutex_);  // read-only: shared lock
    int32_t count = 0;
    std::vector<const RadixNode*> stack = {root_.get()};
    while (!stack.empty()) {
        const RadixNode* node = stack.back();
        stack.pop_back();
        count += (int32_t)node->seq_ids.size();
        for (auto& [key, child] : node->children) {
            stack.push_back(child.get());
        }
    }
    return count;
}

// --- JSON serialization ---

static std::string node_to_json(const RadixNode* node, int32_t cumulative_tokens) {
    std::string s = "{";
    int32_t depth = cumulative_tokens + (int32_t)node->tokens.size();
    s += "\"token_count\":"       + std::to_string(node->tokens.size()) + ",";
    s += "\"cumulative_tokens\":" + std::to_string(depth) + ",";
    s += "\"ref_count\":"         + std::to_string(node->ref_count) + ",";
    s += "\"canonical_len\":"     + std::to_string(node->canonical_len) + ",";
    s += "\"last_access_tick\":"  + std::to_string(node->last_access_tick) + ",";
    s += "\"seq_ids\":[";
    bool first = true;
    for (auto sid : node->seq_ids) {
        if (!first) s += ",";
        s += std::to_string(sid);
        first = false;
    }
    s += "],\"ready_seq_ids\":[";
    first = true;
    for (auto sid : node->ready_seq_ids) {
        if (!first) s += ",";
        s += std::to_string(sid);
        first = false;
    }
    s += "],\"children\":[";
    bool first_child = true;
    for (const auto& [key, child] : node->children) {
        if (!first_child) s += ",";
        s += node_to_json(child.get(), depth);
        first_child = false;
    }
    s += "]}";
    return s;
}

std::string RadixTree::to_json() const {
    std::shared_lock<std::shared_mutex> lock(mutex_);

    // Count active sequences without re-entering the lock
    int32_t active = 0;
    std::vector<const RadixNode*> stk = {root_.get()};
    while (!stk.empty()) {
        const RadixNode* n = stk.back(); stk.pop_back();
        active += (int32_t)n->seq_ids.size();
        for (const auto& [k, c] : n->children) stk.push_back(c.get());
    }

    std::string out = "{\"tick\":" + std::to_string(tick_) + ",";
    out += "\"active_sequences\":" + std::to_string(active) + ",";
    out += "\"root\":" + node_to_json(root_.get(), 0) + "}";
    return out;
}

} // namespace radixforge
