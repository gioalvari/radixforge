#pragma once

#include <atomic>
#include <cstdint>

namespace radixforge {

// Created only when --debug-metrics is enabled. Keeping the pointer null in
// normal operation avoids clock reads and atomic operations on the hot path.
struct PerfMetrics {
    std::atomic<uint64_t> queue_wait_us{0};
    std::atomic<uint64_t> queue_wait_requests{0};
    std::atomic<uint64_t> tokenize_us{0};
    std::atomic<uint64_t> tokenize_calls{0};
    std::atomic<uint64_t> cache_lookup_us{0};
    std::atomic<uint64_t> cache_lookup_calls{0};
    std::atomic<uint64_t> cache_insert_us{0};
    std::atomic<uint64_t> cache_insert_calls{0};
    std::atomic<uint64_t> prefill_decode_us{0};
    std::atomic<uint64_t> prefill_decode_tokens{0};
    std::atomic<uint64_t> generation_decode_us{0};
    std::atomic<uint64_t> generation_decode_tokens{0};
    std::atomic<uint64_t> sampler_us{0};
    std::atomic<uint64_t> sampler_calls{0};
    std::atomic<uint64_t> callback_us{0};
    std::atomic<uint64_t> callback_calls{0};
    std::atomic<uint64_t> sse_write_us{0};
    std::atomic<uint64_t> sse_writes{0};
};

} // namespace radixforge
