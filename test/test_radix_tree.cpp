// Unit tests for RadixTree — no llama.cpp runtime, only type definitions.
// Run: ./test_radix_tree  (prints PASS/FAIL per test, exits 0 on success)

#include "radixforge/radix_tree.h"

#include <cstdio>
#include <cstdlib>
#include <string>
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

// ─── helpers ────────────────────────────────────────────────────────────────

static std::vector<llama_token> tok(std::initializer_list<int> il) {
    return std::vector<llama_token>(il.begin(), il.end());
}

// ─── tests ──────────────────────────────────────────────────────────────────

static void test_empty_insert() {
    printf("\n[test_empty_insert]\n");
    radixforge::RadixTree tree;
    auto m = tree.insert(tok({1, 2, 3, 4, 5}));
    CHECK(m.matched_tokens == 0,     "empty tree: no prefix matched");
    CHECK(m.remaining.size() == 5,   "empty tree: all 5 tokens in remaining");
    CHECK(m.remaining[0] == 1,       "remaining[0] == 1");
    CHECK(m.remaining[4] == 5,       "remaining[4] == 5");
}

static void test_exact_repeat() {
    printf("\n[test_exact_repeat]\n");
    radixforge::RadixTree tree;
    tree.insert(tok({10, 20, 30}));
    auto m = tree.insert(tok({10, 20, 30}));
    CHECK(m.matched_tokens == 3,     "exact repeat: full match (3 tokens)");
    CHECK(m.remaining.empty(),       "exact repeat: no remaining tokens");
}

static void test_partial_match() {
    printf("\n[test_partial_match]\n");
    radixforge::RadixTree tree;
    tree.insert(tok({1, 2, 3, 4, 5}));
    auto m = tree.insert(tok({1, 2, 3, 6, 7}));
    CHECK(m.matched_tokens == 3,     "partial match: 3 shared tokens");
    CHECK(m.remaining.size() == 2,   "partial match: 2 remaining tokens");
    CHECK(m.remaining[0] == 6,       "remaining[0] == 6");
    CHECK(m.remaining[1] == 7,       "remaining[1] == 7");
}

static void test_split_creates_shared_prefix() {
    printf("\n[test_split_creates_shared_prefix]\n");
    radixforge::RadixTree tree;
    // Insert [1,2,3,4] — creates leaf with all 4 tokens
    tree.insert(tok({1, 2, 3, 4}));
    // Insert [1,2,5,6] — diverges after token 2; tree must split at pos 2
    auto m = tree.insert(tok({1, 2, 5, 6}));
    CHECK(m.matched_tokens == 2,     "split: 2 tokens in shared prefix");
    CHECK(m.remaining.size() == 2,   "split: 2 diverging tokens remaining");
    CHECK(m.remaining[0] == 5,       "remaining[0] == 5 (diverge point)");

    // A third insert with the same prefix should also get matched_tokens >= 2
    auto m2 = tree.insert(tok({1, 2, 9, 9}));
    CHECK(m2.matched_tokens >= 2,    "third insert also matches prefix node");
}

static void test_full_match_after_split() {
    printf("\n[test_full_match_after_split]\n");
    radixforge::RadixTree tree;
    tree.insert(tok({1, 2, 3}));
    tree.insert(tok({1, 2, 4}));
    // Re-inserting the first sequence should still give a full match
    auto m = tree.insert(tok({1, 2, 3}));
    CHECK(m.matched_tokens == 3,     "full match after split: matched_tokens == 3");
    CHECK(m.remaining.empty(),       "full match after split: no remaining");
}

static void test_active_sequence_count() {
    printf("\n[test_active_sequence_count]\n");
    radixforge::RadixTree tree;
    CHECK(tree.active_sequence_count() == 0, "initial count: 0");

    auto m = tree.insert(tok({1, 2, 3}));
    // No physical seq_id assigned yet (that's KVMapper's job) so count is still 0
    CHECK(tree.active_sequence_count() == 0, "no seq_ids yet: still 0");

    m.node->seq_ids.insert(42);
    CHECK(tree.active_sequence_count() == 1, "after inserting seq_id 42: count == 1");

    m.node->seq_ids.insert(99);
    CHECK(tree.active_sequence_count() == 2, "after inserting seq_id 99: count == 2");
}

static void test_release_decrements_ref_count() {
    printf("\n[test_release_decrements_ref_count]\n");
    radixforge::RadixTree tree;
    auto m = tree.insert(tok({5, 6, 7}));
    int before = m.node->ref_count;
    CHECK(before == 1,               "insert increments ref_count to 1");
    tree.release(m.node);
    CHECK(m.node->ref_count == 0,    "release decrements ref_count to 0");
}

static void test_eviction_candidates() {
    printf("\n[test_eviction_candidates]\n");
    radixforge::RadixTree tree;

    auto m1 = tree.insert(tok({1, 2, 3}));
    auto m2 = tree.insert(tok({4, 5, 6}));
    auto m3 = tree.insert(tok({7, 8, 9}));

    // Release all so ref_count == 0 → candidates for eviction
    tree.release(m1.node);
    tree.release(m2.node);
    tree.release(m3.node);

    auto candidates = tree.find_eviction_candidates(2);
    CHECK(candidates.size() == 2,    "find_eviction_candidates(2): returns exactly 2");
    CHECK(candidates[0] != nullptr,  "candidate[0] is non-null");
    CHECK(candidates[1] != nullptr,  "candidate[1] is non-null");
    CHECK(candidates[0] != candidates[1], "candidate[0] != candidate[1]");

    // Request more than available
    auto all = tree.find_eviction_candidates(10);
    CHECK(all.size() == 3,           "find_eviction_candidates(10): capped at 3 leaves");
}

static void test_to_json() {
    printf("\n[test_to_json]\n");
    radixforge::RadixTree tree;
    tree.insert(tok({1, 2, 3}));
    tree.insert(tok({1, 2, 4}));

    std::string j = tree.to_json();
    CHECK(!j.empty(),                        "to_json: non-empty");
    CHECK(j.find("\"tick\"") != std::string::npos,   "to_json: contains 'tick'");
    CHECK(j.find("\"root\"") != std::string::npos,   "to_json: contains 'root'");
    CHECK(j.find("\"children\"") != std::string::npos, "to_json: contains 'children'");
    // Verify the JSON starts/ends like an object
    CHECK(j.front() == '{',                  "to_json: starts with '{'");
    CHECK(j.back() == '}',                   "to_json: ends with '}'");
}

// ─── main ────────────────────────────────────────────────────────────────────

int main() {
    printf("=== RadixTree Unit Tests ===\n");

    test_empty_insert();
    test_exact_repeat();
    test_partial_match();
    test_split_creates_shared_prefix();
    test_full_match_after_split();
    test_active_sequence_count();
    test_release_decrements_ref_count();
    test_eviction_candidates();
    test_to_json();

    printf("\n─────────────────────────────\n");
    printf("Results: %d passed, %d failed\n", g_pass, g_fail);
    if (g_fail > 0) {
        printf("FAIL\n");
        return 1;
    }
    printf("PASS\n");
    return 0;
}
