// llama.cpp-backed KvOps adapter. Kept out of kv_mapper.cpp so the mapper can be
// unit-tested against a fake KvOps without linking llama.cpp.
#include "radixforge/kv_mapper.h"

#include <memory>

namespace radixforge {

void LlamaKvOps::seq_cp(llama_seq_id src, llama_seq_id dst, int32_t p0,
                        int32_t p1) {
    bridge_.memory_seq_cp(src, dst, p0, p1);
}

void LlamaKvOps::seq_rm(llama_seq_id seq_id, int32_t p0, int32_t p1) {
    bridge_.memory_seq_rm(seq_id, p0, p1);
}

KVMapper::KVMapper(LlamaBridge& bridge, RadixTree& tree, const Config& config)
    : owned_kv_ops_(std::make_unique<LlamaKvOps>(bridge)),
      kv_ops_(*owned_kv_ops_), tree_(tree), config_(config) {
    // Initialize free seq_id pool (0 is reserved for scratch)
    for (int32_t i = 1; i < config.max_sequences; i++) {
        free_seq_ids_.push_back(i);
    }
}

} // namespace radixforge
