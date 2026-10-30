#pragma once

#include "radixforge/config.h"
#include "llama.h"

#include <memory>
#include <string>
#include <vector>

namespace radixforge {

// RAII wrappers for llama.cpp resources
struct LlamaModelDeleter {
    void operator()(llama_model* m) const;
};

struct LlamaContextDeleter {
    void operator()(llama_context* c) const;
};

using LlamaModelPtr = std::unique_ptr<llama_model, LlamaModelDeleter>;
using LlamaContextPtr = std::unique_ptr<llama_context, LlamaContextDeleter>;

class LlamaBridge {
public:
    explicit LlamaBridge(const Config& config);
    ~LlamaBridge();

    // Non-copyable, non-movable
    LlamaBridge(const LlamaBridge&) = delete;
    LlamaBridge& operator=(const LlamaBridge&) = delete;
    LlamaBridge(LlamaBridge&&) = delete;
    LlamaBridge& operator=(LlamaBridge&&) = delete;

    // Accessors
    llama_model* model() const { return model_.get(); }
    llama_context* ctx() const { return ctx_.get(); }
    const llama_vocab* vocab() const { return vocab_; }

    // Apply the model's native chat template (Jinja/ChatML embedded in GGUF).
    // Falls back to ChatML if the model has no embedded template.
    // Messages: vector of (role, content) pairs.
    std::string apply_chat_template(
        const std::vector<std::pair<std::string, std::string>>& messages) const;

    // Tokenize a string into token IDs
    std::vector<llama_token> tokenize(const std::string& text, bool add_bos = true) const;

    // Memory (KV cache) operations — uses llama_memory_* API
    void memory_seq_cp(llama_seq_id src, llama_seq_id dst, int32_t p0, int32_t p1);
    void memory_seq_rm(llama_seq_id seq_id, int32_t p0, int32_t p1);

    // Per-request sampler management.
    // Caller owns the returned pointer and must call free_sampler() when done.
    llama_sampler* create_sampler(float temperature, float top_p) const;
    static void free_sampler(llama_sampler* sampler);

    int32_t n_ctx() const;
    int32_t n_batch() const;
    std::string model_name() const;  // human-readable model name from GGUF metadata

private:
    Config config_;
    LlamaModelPtr model_;
    LlamaContextPtr ctx_;
    const llama_vocab* vocab_ = nullptr;
};

} // namespace radixforge
