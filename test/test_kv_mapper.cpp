// Unit tests for KVMapper lifecycle and KV operation ranges without a model.

#include "radixforge/kv_mapper.h"

#include <cstdio>
#include <cstdlib>
#include <vector>

static int g_pass = 0;
static int g_fail = 0;

#define CHECK(cond, msg) \
    do { \
        if (cond) { \
            ++g_pass; \
            printf("  PASS  %s\n", msg); \
        } else { \
            ++g_fail; \
            printf("  FAIL  %s  (line %d)\n", msg, __LINE__); \
        } \
    } while (0)

struct Call {
    enum class Type { Copy, Remove } type;
    llama_seq_id src;
    llama_seq_id dst;
    int32_t p0;
    int32_t p1;
};

class RecordingKvOps final : public radixforge::KvOps {
public:
    void seq_cp(llama_seq_id src, llama_seq_id dst, int32_t p0, int32_t p1) override {
        calls.push_back({Call::Type::Copy, src, dst, p0, p1});
    }

    void seq_rm(llama_seq_id seq_id, int32_t p0, int32_t p1) override {
        calls.push_back({Call::Type::Remove, seq_id, -1, p0, p1});
    }

    int remove_count(llama_seq_id seq_id, int32_t p0, int32_t p1) const {
        int count = 0;
        for (const auto& call : calls) {
            if (call.type == Call::Type::Remove && call.src == seq_id &&
                call.p0 == p0 && call.p1 == p1) {
                ++count;
            }
        }
        return count;
    }

    std::vector<Call> calls;
};

static std::vector<llama_token> tok(std::initializer_list<int> values) {
    return std::vector<llama_token>(values.begin(), values.end());
}

static radixforge::Config config_with_sequences(int32_t max_sequences) {
    radixforge::Config config;
    config.max_sequences = max_sequences;
    return config;
}

static void test_retained_sequence_is_trimmed() {
    printf("\n[test_retained_sequence_is_trimmed]\n");
    RecordingKvOps ops;
    radixforge::RadixTree tree;
    radixforge::KVMapper mapper(ops, tree, config_with_sequences(3));

    auto prepared = mapper.prepare_sequence(tok({1, 2, 3}));
    mapper.mark_sequence_ready(prepared.seq_id, prepared.leaf_node);
    mapper.release_sequence(prepared.seq_id, prepared.leaf_node);
    tree.release(prepared.leaf_node);

    CHECK(ops.remove_count(prepared.seq_id, 3, -1) == 1,
          "retained canonical sequence is trimmed to canonical prompt length");
    CHECK(tree.active_sequence_count() == 1,
          "sole ready sequence remains retained as the cache source");
}

static void test_full_cache_hit_decodes_final_prompt_token() {
    printf("\n[test_full_cache_hit_decodes_final_prompt_token]\n");
    RecordingKvOps ops;
    radixforge::RadixTree tree;
    radixforge::KVMapper mapper(ops, tree, config_with_sequences(3));
    const auto prompt = tok({10, 20, 30});

    auto original = mapper.prepare_sequence(prompt);
    mapper.mark_sequence_ready(original.seq_id, original.leaf_node);
    mapper.release_sequence(original.seq_id, original.leaf_node);
    tree.release(original.leaf_node);

    auto hit = mapper.prepare_sequence(prompt);
    CHECK(hit.cached_pos == 2, "full hit copies only the first n-1 KV positions");
    CHECK(hit.delta.size() == 1 && hit.delta[0] == 30,
          "full hit returns a non-empty delta containing the last prompt token");
    CHECK(!ops.calls.empty() && ops.calls.back().type == Call::Type::Copy &&
              ops.calls.back().p0 == 0 && ops.calls.back().p1 == 2,
          "KV copy range ends before the token decoded for logits");

    mapper.release_sequence(hit.seq_id, hit.leaf_node);
    tree.release(hit.leaf_node);
}

static void test_double_release_does_not_duplicate_free_id() {
    printf("\n[test_double_release_does_not_duplicate_free_id]\n");
    RecordingKvOps ops;
    radixforge::RadixTree tree;
    radixforge::KVMapper mapper(ops, tree, config_with_sequences(2));

    auto first = mapper.prepare_sequence(tok({1}));
    mapper.release_sequence(first.seq_id, first.leaf_node);
    mapper.release_sequence(first.seq_id, first.leaf_node);
    tree.release(first.leaf_node);
    CHECK(ops.remove_count(first.seq_id, 0, -1) == 1,
          "double release clears the physical sequence exactly once");

    auto second = mapper.prepare_sequence(tok({2}));
    CHECK(second.seq_id == first.seq_id, "released id is reused once");
    mapper.release_sequence(second.seq_id, second.leaf_node);
    tree.release(second.leaf_node);
}

static void test_exhaustion_releases_insert_reference() {
    printf("\n[test_exhaustion_releases_insert_reference]\n");
    RecordingKvOps ops;
    radixforge::RadixTree tree;
    radixforge::KVMapper mapper(ops, tree, config_with_sequences(2));

    auto occupied = mapper.prepare_sequence(tok({1, 2}));
    bool exhausted = false;
    try {
        (void)mapper.prepare_sequence(tok({9, 9}));
    } catch (const radixforge::SequenceCapacityExhausted&) {
        exhausted = true;
    }
    CHECK(exhausted, "sequence pool reports exhaustion when no idle cache exists");

    mapper.release_sequence(occupied.seq_id, occupied.leaf_node);
    tree.release(occupied.leaf_node);
    auto reusable = mapper.prepare_sequence(tok({9, 9}));
    CHECK(reusable.seq_id == occupied.seq_id,
          "exhausted preparation does not leak a tree reference or sequence id");
    mapper.release_sequence(reusable.seq_id, reusable.leaf_node);
    tree.release(reusable.leaf_node);
}

static void test_generated_tokens_extend_retained_cache() {
    printf("\n[test_generated_tokens_extend_retained_cache]\n");
    RecordingKvOps ops;
    radixforge::RadixTree tree;
    radixforge::KVMapper mapper(ops, tree, config_with_sequences(3));

    const auto prompt = tok({1, 2, 3});
    auto prepared = mapper.prepare_sequence(prompt);
    mapper.mark_sequence_ready(prepared.seq_id, prepared.leaf_node);
    const auto with_generated = tok({1, 2, 3, 4, 5});
    radixforge::RadixNode* extended =
        mapper.extend_sequence(prepared.seq_id, prepared.leaf_node, with_generated);
    mapper.release_sequence(prepared.seq_id, extended);
    tree.release(prepared.leaf_node);

    const auto follow_up = tok({1, 2, 3, 4, 5, 6, 7});
    auto next = mapper.prepare_sequence(follow_up);
    CHECK(next.cached_pos == 5,
          "follow-up copies the full prompt plus decoded generated tokens");
    CHECK(next.delta.size() == 2 && next.delta[0] == 6,
          "follow-up decodes only tokens beyond the retained generated prefix");
    CHECK(!ops.calls.empty() && ops.calls.back().type == Call::Type::Copy &&
              ops.calls.back().p1 == 5,
          "follow-up KV copy uses the full generated-token cache coverage");

    mapper.release_sequence(next.seq_id, next.leaf_node);
    tree.release(next.leaf_node);
}

static void test_high_watermark_evicts_idle_retained_sequences() {
    printf("\n[test_high_watermark_evicts_idle_retained_sequences]\n");
    RecordingKvOps ops;
    radixforge::Config config = config_with_sequences(4);
    config.n_ctx = 10;
    config.kv_high_watermark = 0.5f;
    radixforge::RadixTree tree;
    radixforge::KVMapper mapper(ops, tree, config);

    auto first = mapper.prepare_sequence(tok({1, 2, 3}));
    mapper.mark_sequence_ready(first.seq_id, first.leaf_node);
    mapper.release_sequence(first.seq_id, first.leaf_node);
    tree.release(first.leaf_node);
    auto second = mapper.prepare_sequence(tok({4, 5, 6}));
    mapper.mark_sequence_ready(second.seq_id, second.leaf_node);
    mapper.release_sequence(second.seq_id, second.leaf_node);
    tree.release(second.leaf_node);
    CHECK(mapper.kv_cells_used() == 6, "two retained prefixes consume six KV cells");

    auto third = mapper.prepare_sequence(tok({7}));
    CHECK(mapper.metrics().evictions.load() == 1,
          "preparation evicts the LRU idle retained sequence above watermark");
    CHECK(mapper.kv_cells_used() <= 3,
          "eviction reduces tracked KV usage below the high watermark");
    CHECK(ops.remove_count(first.seq_id, 0, -1) == 1,
          "watermark eviction clears the oldest retained sequence");
    mapper.release_sequence(third.seq_id, third.leaf_node);
    tree.release(third.leaf_node);
}

int main() {
    printf("=== KVMapper Unit Tests ===\n");
    test_retained_sequence_is_trimmed();
    test_full_cache_hit_decodes_final_prompt_token();
    test_double_release_does_not_duplicate_free_id();
    test_exhaustion_releases_insert_reference();
    test_generated_tokens_extend_retained_cache();
    test_high_watermark_evicts_idle_retained_sequences();

    printf("\nResults: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
