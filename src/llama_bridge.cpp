#include "radixforge/llama_bridge.h"

#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

namespace radixforge {

void LlamaModelDeleter::operator()(llama_model* m) const {
    if (m) llama_model_free(m);
}

void LlamaContextDeleter::operator()(llama_context* c) const {
    if (c) llama_free(c);
}

LlamaBridge::LlamaBridge(const Config& config) : config_(config) {
    llama_backend_init();

    auto model_params = llama_model_default_params();
    model_params.n_gpu_layers = config.n_gpu_layers;

    model_.reset(llama_model_load_from_file(config.model_path.c_str(), model_params));
    if (!model_) {
        throw std::runtime_error("Failed to load model: " + config.model_path);
    }

    vocab_ = llama_model_get_vocab(model_.get());

    auto ctx_params = llama_context_default_params();
    ctx_params.n_ctx     = config.n_ctx;
    ctx_params.n_batch   = config.n_batch;
    ctx_params.n_ubatch  = config.n_batch;  // must equal n_batch for encoder models
    ctx_params.n_seq_max = config.max_sequences;

    ctx_.reset(llama_init_from_model(model_.get(), ctx_params));
    if (!ctx_) {
        throw std::runtime_error("Failed to create llama context");
    }

    init_sampler(0.8f, 0.95f);
}

LlamaBridge::~LlamaBridge() {
    if (sampler_) {
        llama_sampler_free(sampler_);
        sampler_ = nullptr;
    }
    ctx_.reset();
    model_.reset();
    llama_backend_free();
}

void LlamaBridge::init_sampler(float temperature, float top_p) {
    sampler_ = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(sampler_, llama_sampler_init_temp(temperature));
    llama_sampler_chain_add(sampler_, llama_sampler_init_top_p(top_p, 1));
    llama_sampler_chain_add(sampler_, llama_sampler_init_dist(0));
}

void LlamaBridge::reset_sampler(float temperature, float top_p) {
    if (sampler_) {
        llama_sampler_free(sampler_);
        sampler_ = nullptr;
    }
    init_sampler(temperature, top_p);
}

std::vector<llama_token> LlamaBridge::tokenize(const std::string& text, bool add_bos) const {
    std::vector<llama_token> tokens(text.size() + 16);
    int32_t n = llama_tokenize(vocab_, text.c_str(), text.size(),
                               tokens.data(), static_cast<int32_t>(tokens.size()),
                               add_bos, true);
    if (n < 0) {
        tokens.resize(-n);
        n = llama_tokenize(vocab_, text.c_str(), text.size(),
                           tokens.data(), static_cast<int32_t>(tokens.size()),
                           add_bos, true);
    }
    tokens.resize(n);
    return tokens;
}

bool LlamaBridge::decode(const std::vector<llama_token>& tokens, llama_seq_id seq_id,
                         int32_t pos_start) {
    const int32_t n_tokens = static_cast<int32_t>(tokens.size());
    llama_batch batch = llama_batch_init(n_tokens, 0, 1);
    batch.n_tokens = n_tokens;

    for (int32_t i = 0; i < n_tokens; i++) {
        batch.token[i]      = tokens[i];
        batch.pos[i]        = pos_start + i;
        batch.n_seq_id[i]   = 1;
        batch.seq_id[i][0]  = seq_id;  // write INTO the allocated slot, not overwrite pointer
        batch.logits[i]     = (i == n_tokens - 1) ? 1 : 0;
    }

    int status = llama_decode(ctx_.get(), batch);
    llama_batch_free(batch);

    if (status != 0) {
        fprintf(stderr, "[radixforge] llama_decode failed with status %d\n", status);
        return false;
    }
    return true;
}

bool LlamaBridge::decode_single(llama_token token, llama_seq_id seq_id, int32_t pos) {
    llama_batch batch = llama_batch_init(1, 0, 1);
    batch.n_tokens = 1;

    batch.token[0]      = token;
    batch.pos[0]        = pos;
    batch.n_seq_id[0]   = 1;
    batch.seq_id[0][0]  = seq_id;  // write INTO the allocated slot
    batch.logits[0]     = 1;

    int status = llama_decode(ctx_.get(), batch);
    llama_batch_free(batch);

    if (status != 0) {
        fprintf(stderr, "[radixforge] decode_single failed with status %d\n", status);
        return false;
    }
    return true;
}

llama_token LlamaBridge::sample() {
    return llama_sampler_sample(sampler_, ctx_.get(), -1);
}

void LlamaBridge::memory_seq_cp(llama_seq_id src, llama_seq_id dst,
                                 int32_t p0, int32_t p1) {
    llama_memory_t mem = llama_get_memory(ctx_.get());
    llama_memory_seq_cp(mem, src, dst, p0, p1);
}

void LlamaBridge::memory_seq_rm(llama_seq_id seq_id, int32_t p0, int32_t p1) {
    llama_memory_t mem = llama_get_memory(ctx_.get());
    llama_memory_seq_rm(mem, seq_id, p0, p1);
}

int32_t LlamaBridge::n_ctx() const {
    return llama_n_ctx(ctx_.get());
}

int32_t LlamaBridge::n_batch() const {
    return llama_n_batch(ctx_.get());
}

std::string LlamaBridge::apply_chat_template(
    const std::vector<std::pair<std::string, std::string>>& messages) const
{
    std::vector<llama_chat_message> chat;
    chat.reserve(messages.size());
    for (const auto& [role, content] : messages) {
        chat.push_back({role.c_str(), content.c_str()});
    }

    // Retrieve the chat template embedded in the GGUF model metadata.
    // Returns nullptr if the model has no embedded template.
    const char* tmpl = llama_model_chat_template(model_.get(), nullptr);

    // First call: measure required buffer size
    int32_t n = llama_chat_apply_template(tmpl, chat.data(), chat.size(),
                                          /*add_ass=*/true, nullptr, 0);
    if (n > 0) {
        std::string result(n, '\0');
        llama_chat_apply_template(tmpl, chat.data(), chat.size(),
                                  /*add_ass=*/true, result.data(), n);
        return result;
    }

    // Fallback: manual ChatML (works for Llama-3 / Qwen / Mistral-instruct)
    fprintf(stderr, "[radixforge] No embedded chat template — using ChatML fallback\n");
    std::string result;
    for (const auto& [role, content] : messages) {
        result += "<|im_start|>" + role + "\n" + content + "<|im_end|>\n";
    }
    result += "<|im_start|>assistant\n";
    return result;
}

llama_sampler* LlamaBridge::create_sampler(float temperature, float top_p) const {
    llama_sampler* s = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(s, llama_sampler_init_temp(temperature));
    llama_sampler_chain_add(s, llama_sampler_init_top_p(top_p, 1));
    llama_sampler_chain_add(s, llama_sampler_init_dist(0));
    return s;
}

void LlamaBridge::free_sampler(llama_sampler* sampler) {
    if (sampler) llama_sampler_free(sampler);
}

} // namespace radixforge
