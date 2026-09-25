#pragma once

#include <cstdint>
#include <string>

namespace radixforge {

struct Config {
    // Model
    std::string model_path;
    int32_t n_gpu_layers = 99;      // offload all layers to Metal
    int32_t n_ctx        = 32768;   // max KV cache size in tokens
    int32_t n_batch      = 2048;    // max batch size for decode

    // Radix Tree / Sequences
    int32_t max_sequences = 32;     // max concurrent llama_seq_id
    int32_t gc_watermark  = 28;     // trigger LRU eviction above this

    // Server
    std::string host = "127.0.0.1";
    int32_t port     = 8400;
    int32_t n_threads_http = 4;
    int32_t admit_window_ms = 2;  // idle-only request coalescing window

    // Generation defaults
    int32_t max_tokens   = 4096;
    float   temperature  = 0.7f;
    float   top_p        = 0.9f;

    // Security
    std::string admin_token;  // if set, /admin/* requires Authorization: Bearer <token>

    // Logging
    std::string log_level = "info";  // debug | info | warn | error
};

} // namespace radixforge
