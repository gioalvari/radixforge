#include "radixforge/server.h"
#include "radixforge/logger.h"

#include <algorithm>
#include <chrono>
#include <iterator>

namespace radixforge {

// --- InferenceWorker ---

InferenceWorker::InferenceWorker(LlamaBridge& bridge, RadixTree& tree,
                                 KVMapper& mapper, const Config& config)
    : bridge_(bridge), tree_(tree), mapper_(mapper),
      admit_window_ms_(config.admit_window_ms) {}

InferenceWorker::~InferenceWorker() { stop(); }

void InferenceWorker::start() {
    running_.store(true);
    worker_thread_ = std::thread(&InferenceWorker::run, this);
}

void InferenceWorker::stop() {
    running_.store(false);
    queue_cv_.notify_all();
    if (worker_thread_.joinable()) worker_thread_.join();
}

void InferenceWorker::enqueue(InferenceRequest req) {
    req.admitted_at = std::chrono::steady_clock::now();
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        queue_.push_back(std::move(req));
    }
    queue_cv_.notify_one();
}

size_t InferenceWorker::pending_count() const {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    return queue_.size();
}

int32_t InferenceWorker::active_sequence_count() const {
    return tree_.active_sequence_count();
}

int32_t InferenceWorker::generating_sequence_count() const {
    return generating_sequences_.load(std::memory_order_relaxed);
}

bool InferenceWorker::is_cancelled(const InferenceRequest& req) {
    return req.cancelled && req.cancelled();
}

// ── Phase 1: tokenise + KV-prepare every request ─────────────────────────────
// No llama_decode yet — just builds PrefillCtx for each request.
std::vector<PrefillCtx> InferenceWorker::prepare_all(
    std::vector<InferenceRequest>& reqs, std::vector<InferenceRequest>* deferred,
    std::vector<InferenceRequest>* queued) {

    std::vector<PrefillCtx> ctxs;
    ctxs.reserve(reqs.size());

    for (size_t req_index = 0; req_index < reqs.size(); ++req_index) {
        auto& req = reqs[req_index];
        if (is_cancelled(req)) continue;
        // Resolve chat template
        std::string prompt = req.prompt;
        if (prompt.empty() && !req.messages.empty()) {
            std::vector<std::pair<std::string,std::string>> chat_msgs;
            chat_msgs.reserve(req.messages.size());
            for (const auto& m : req.messages)
                chat_msgs.push_back({m.role, m.content});
            prompt = bridge_.apply_chat_template(chat_msgs);
        }

        std::vector<llama_token> tokens = bridge_.tokenize(prompt, /*add_bos=*/true);

        // Guard: clamp max_tokens so the total never exceeds the context window.
        int32_t available = bridge_.n_ctx() - (int32_t)tokens.size() - 4;
        if (available <= 0) {
            RF_WARN("worker", "Prompt too long (%zu tokens) for ctx %d — rejected",
                     tokens.size(), bridge_.n_ctx());
            if (req.on_error) req.on_error("prompt exceeds context window", 400);
            continue;
        }
        if (req.max_tokens > available) {
            RF_DEBUG("worker", "max_tokens clamped %d -> %d (ctx limit)",
                     req.max_tokens, available);
            req.max_tokens = available;
        }

        const PrefixAvailability availability = mapper_.inspect_prefix(tokens);
        const int32_t pending_advantage =
            availability.pending_tokens - availability.ready_tokens;
        if (pending_advantage > 0) {
            // Do not insert/allocate this request yet: the leading pending
            // request will become ready (or be released on failure) this run.
            // This prevents a copy from empty KV while retaining prefix sharing.
            if (deferred) deferred->push_back(std::move(req));
            continue;
        }

        PrefillCtx ctx;
        ctx.req = std::move(req);
        ctx.tokens = std::move(tokens);

        try {
            ctx.prep  = mapper_.prepare_sequence(ctx.tokens);
        } catch (const SequenceCapacityExhausted&) {
            if (queued) {
                queued->push_back(std::move(ctx.req));
                for (++req_index; req_index < reqs.size(); ++req_index) {
                    if (!is_cancelled(reqs[req_index])) {
                        queued->push_back(std::move(reqs[req_index]));
                    }
                }
            }
            break;
        } catch (const std::exception& e) {
            RF_ERROR("worker", "prepare_sequence failed: %s", e.what());
            if (ctx.req.on_error) ctx.req.on_error("resource exhausted", 503);
            continue;
        }

        if (ctx.prep.delta.empty()) {
            RF_ERROR("worker", "prepare_sequence returned an empty delta");
            mapper_.release_sequence(ctx.prep.seq_id, ctx.prep.leaf_node);
            tree_.release(ctx.prep.leaf_node);
            if (ctx.req.on_error) {
                ctx.req.on_error("internal cache preparation error", 500);
            }
            continue;
        }
        ctx.delta_start = 0;
        ctx.stats.prompt_n = static_cast<int32_t>(ctx.prep.delta.size());
        ctx.stats.cache_n = ctx.prep.cached_pos;
        ctx.stats.prompt_tokens = static_cast<int32_t>(ctx.tokens.size());

        RF_DEBUG("worker", "Prefill: %zu tokens, cached: %d, delta: %zu",
                 ctx.tokens.size(), ctx.prep.cached_pos, ctx.prep.delta.size());

        ctxs.push_back(std::move(ctx));
    }
    return ctxs;
}

void InferenceWorker::defer_to_front(std::vector<InferenceRequest>& deferred) {
    if (deferred.empty()) return;
    std::lock_guard<std::mutex> lock(queue_mutex_);
    for (auto it = deferred.rbegin(); it != deferred.rend(); ++it) {
        queue_.push_front(std::move(*it));
    }
    queue_cv_.notify_one();
}

// ── Fused continuous-batch step ─────────────────────────────────────────────
void InferenceWorker::retire(ActiveSeq& seq, bool notify_client) {
    RadixNode* sequence_node = seq.leaf_node;
    if (notify_client && !is_cancelled(seq.req)) {
        sequence_node = mapper_.extend_sequence(seq.seq_id, seq.leaf_node,
                                                seq.cached_tokens);
    }
    if (notify_client && seq.req.on_token && !is_cancelled(seq.req)) {
        const auto finished = std::chrono::steady_clock::now();
        seq.stats.predicted_ms = std::chrono::duration<double, std::milli>(
            finished - seq.generation_started_at).count();
        seq.req.on_token("", true, &seq.stats);
    }
    LlamaBridge::free_sampler(seq.sampler);
    mapper_.release_sequence(seq.seq_id, sequence_node);
    tree_.release(seq.leaf_node);
}

void InferenceWorker::fused_step(std::vector<ActiveSeq>& seqs,
                                 std::vector<PrefillCtx>& prefilling,
                                 size_t newly_admitted) {
    for (auto& seq : seqs) {
        if (is_cancelled(seq.req)) seq.done = true;
    }
    auto cancelled = std::remove_if(seqs.begin(), seqs.end(), [this](ActiveSeq& seq) {
        if (!seq.done) return false;
        RF_INFO("worker", "Retiring cancelled sequence %d", seq.seq_id);
        retire(seq, false);
        return true;
    });
    seqs.erase(cancelled, seqs.end());

    auto abandoned = std::remove_if(prefilling.begin(), prefilling.end(), [this](PrefillCtx& ctx) {
        if (!is_cancelled(ctx.req)) return false;
        mapper_.release_sequence(ctx.prep.seq_id, ctx.prep.leaf_node);
        tree_.release(ctx.prep.leaf_node);
        return true;
    });
    prefilling.erase(abandoned, prefilling.end());

    const int decoding = static_cast<int>(seqs.size());
    const int32_t capacity = bridge_.n_batch() - decoding;
    if (decoding == 0 && (prefilling.empty() || capacity <= 0)) return;

    llama_batch batch = llama_batch_init(bridge_.n_batch(), 0, 1);
    int batch_idx = 0;
    for (int i = 0; i < decoding; ++i) {
        batch.token[batch_idx] = seqs[i].next_token;
        batch.pos[batch_idx] = seqs[i].pos;
        batch.n_seq_id[batch_idx] = 1;
        batch.seq_id[batch_idx][0] = seqs[i].seq_id;
        batch.logits[batch_idx] = 1;
        ++batch_idx;
    }

    struct PromptSubmission {
        size_t ctx_index;
        int32_t tokens;
        int32_t logit_index;
    };
    std::vector<PromptSubmission> submissions;
    int32_t prompt_tokens = 0;
    for (size_t i = 0; i < prefilling.size() && prompt_tokens < capacity; ++i) {
        auto& ctx = prefilling[i];
        const int32_t total = static_cast<int32_t>(ctx.prep.delta.size());
        const int32_t remaining = total - ctx.delta_start;
        if (remaining <= 0) continue;
        const int32_t take = std::min(remaining, capacity - prompt_tokens);
        const bool completes = take == remaining;
        const int32_t start = ctx.delta_start;
        for (int32_t offset = 0; offset < take; ++offset) {
            const int32_t prompt_index = start + offset;
            batch.token[batch_idx] = ctx.prep.delta[prompt_index];
            batch.pos[batch_idx] = ctx.prep.cached_pos + prompt_index;
            batch.n_seq_id[batch_idx] = 1;
            batch.seq_id[batch_idx][0] = ctx.prep.seq_id;
            batch.logits[batch_idx] = completes && offset == take - 1 ? 1 : 0;
            ++batch_idx;
        }
        submissions.push_back({i, take, completes ? batch_idx - 1 : -1});
        prompt_tokens += take;
    }
    batch.n_tokens = batch_idx;
    if (batch.n_tokens == 0) {
        llama_batch_free(batch);
        return;
    }

    if (newly_admitted > 0) {
        RF_INFO("worker", "step: admitted %zu, prefill %d tokens, decoding %d",
                newly_admitted, prompt_tokens, decoding);
    }

    int decode_status = llama_decode(bridge_.ctx(), batch);
    if (decode_status == 1 && mapper_.evict_idle_sequence()) {
        RF_WARN("worker", "Step stalled by full KV cache; evicted idle cache and retrying");
        decode_status = llama_decode(bridge_.ctx(), batch);
    }
    llama_batch_free(batch);

    auto fail_active = [this](ActiveSeq& seq, const char* message, int status) {
        if (seq.req.on_error && !is_cancelled(seq.req)) {
            seq.req.on_error(message, status);
        }
        LlamaBridge::free_sampler(seq.sampler);
        mapper_.release_sequence(seq.seq_id, seq.leaf_node);
        tree_.release(seq.leaf_node);
    };
    auto fail_prefill = [this](PrefillCtx& ctx, const char* message, int status) {
        if (ctx.req.on_error && !is_cancelled(ctx.req)) {
            ctx.req.on_error(message, status);
        }
        mapper_.release_sequence(ctx.prep.seq_id, ctx.prep.leaf_node);
        tree_.release(ctx.prep.leaf_node);
    };
    if (decode_status == 1) {
        RF_WARN("worker", "Step stalled by full KV cache; failing newest sequence");
        if (!seqs.empty()) {
            fail_active(seqs.back(), "KV cache is full", 503);
            seqs.pop_back();
        } else if (!prefilling.empty()) {
            fail_prefill(prefilling.back(), "KV cache is full", 503);
            prefilling.pop_back();
        }
        return;
    }
    if (decode_status != 0) {
        RF_ERROR("worker", "Fused batch decode failed (%d seqs, %d prompt tokens, status %d)",
                 decoding, prompt_tokens, decode_status);
        for (auto& seq : seqs) fail_active(seq, "inference decode failed", 500);
        seqs.clear();
        for (auto& ctx : prefilling) {
            fail_prefill(ctx, "inference prefill failed", 500);
        }
        prefilling.clear();
        return;
    }

    ++decode_steps_;
    if (prompt_tokens > 0) {
        prefill_tokens_.fetch_add(prompt_tokens, std::memory_order_relaxed);
        prefill_calls_.fetch_add(1, std::memory_order_relaxed);
    }

    // Finish prompt chunks only after the fused decode succeeds.  Pending
    // sequences therefore cannot be copied by a request admitted this step.
    std::vector<int32_t> prompt_logits(prefilling.size(), -1);
    for (const auto& submission : submissions) {
        auto& ctx = prefilling[submission.ctx_index];
        ctx.delta_start += submission.tokens;
        const int32_t total = static_cast<int32_t>(ctx.prep.delta.size());
        if (ctx.delta_start == total) {
            mapper_.mark_sequence_ready(ctx.prep.seq_id, ctx.prep.leaf_node);
            prompt_logits[submission.ctx_index] = submission.logit_index;
        }
    }

    const auto now = std::chrono::steady_clock::now();
    for (int i = 0; i < decoding; ++i) {
        ActiveSeq& seq = seqs[i];
        seq.cached_tokens.push_back(seq.next_token);
        llama_token new_tok = llama_sampler_sample(seq.sampler, bridge_.ctx(), i);
        if (llama_vocab_is_eog(bridge_.vocab(), new_tok)) {
            seq.done = true;
            seq.notify_on_retire = true;
            continue;
        }
        char buf[512];
        const int32_t nc = llama_token_to_piece(bridge_.vocab(), new_tok, buf, sizeof(buf), 0, true);
        std::string piece(buf, nc > 0 ? nc : 0);
        ++seq.stats.predicted_n;
        --seq.remaining;
        const bool last = seq.remaining == 0;
        if (seq.req.on_token && !is_cancelled(seq.req)) {
            if (last) {
                seq.stats.predicted_ms = std::chrono::duration<double, std::milli>(
                    now - seq.generation_started_at).count();
            }
            seq.req.on_token(piece, last, last ? &seq.stats : nullptr);
        }
        if (last || is_cancelled(seq.req)) {
            seq.done = true;
        } else {
            seq.next_token = new_tok;
            ++seq.pos;
        }
    }
    auto completed = std::remove_if(seqs.begin(), seqs.end(), [this](ActiveSeq& seq) {
        if (!seq.done) return false;
        if (is_cancelled(seq.req)) {
            RF_INFO("worker", "Retiring cancelled sequence %d", seq.seq_id);
        }
        retire(seq, seq.notify_on_retire);
        return true;
    });
    seqs.erase(completed, seqs.end());

    std::vector<ActiveSeq> newly_active;
    std::vector<size_t> completed_prefills;
    for (size_t index = 0; index < prefilling.size(); ++index) {
        const int32_t logit_index = prompt_logits[index];
        if (logit_index < 0) continue;
        auto& ctx = prefilling[index];
        if (is_cancelled(ctx.req)) {
            mapper_.release_sequence(ctx.prep.seq_id, ctx.prep.leaf_node);
            tree_.release(ctx.prep.leaf_node);
            completed_prefills.push_back(index);
            continue;
        }
        llama_sampler* sampler = bridge_.create_sampler(ctx.req.temperature, ctx.req.top_p);
        const llama_token first_tok = llama_sampler_sample(sampler, bridge_.ctx(), logit_index);
        ctx.stats.prompt_ms = std::chrono::duration<double, std::milli>(
            now - ctx.req.admitted_at).count();
        if (llama_vocab_is_eog(bridge_.vocab(), first_tok)) {
            if (ctx.req.on_token) ctx.req.on_token("", true, &ctx.stats);
            LlamaBridge::free_sampler(sampler);
            mapper_.release_sequence(ctx.prep.seq_id, ctx.prep.leaf_node);
            tree_.release(ctx.prep.leaf_node);
            completed_prefills.push_back(index);
            continue;
        }
        char buf[512];
        const int32_t nc = llama_token_to_piece(bridge_.vocab(), first_tok, buf,
                                                 sizeof(buf), 0, true);
        std::string piece(buf, nc > 0 ? nc : 0);
        ctx.stats.predicted_n = 1;
        const bool first_is_last = ctx.req.max_tokens <= 1;
        if (first_is_last) ctx.stats.predicted_ms = 0.0;
        if (ctx.req.on_token) {
            ctx.req.on_token(piece, first_is_last, first_is_last ? &ctx.stats : nullptr);
        }
        if (first_is_last) {
            LlamaBridge::free_sampler(sampler);
            mapper_.release_sequence(ctx.prep.seq_id, ctx.prep.leaf_node);
            tree_.release(ctx.prep.leaf_node);
            completed_prefills.push_back(index);
            continue;
        }
        const int32_t end_pos = ctx.prep.cached_pos +
                                static_cast<int32_t>(ctx.prep.delta.size());
        ActiveSeq seq;
        seq.seq_id = ctx.prep.seq_id;
        seq.leaf_node = ctx.prep.leaf_node;
        seq.pos = end_pos;
        seq.next_token = first_tok;
        seq.remaining = ctx.req.max_tokens - 1;
        seq.sampler = sampler;
        seq.stats = ctx.stats;
        seq.req = std::move(ctx.req);
        seq.generation_started_at = now;
        seq.cached_tokens = std::move(ctx.tokens);
        newly_active.push_back(std::move(seq));
        completed_prefills.push_back(index);
    }
    for (auto it = completed_prefills.rbegin(); it != completed_prefills.rend(); ++it) {
        prefilling.erase(prefilling.begin() + static_cast<std::ptrdiff_t>(*it));
    }
    seqs.insert(seqs.end(), std::make_move_iterator(newly_active.begin()),
                std::make_move_iterator(newly_active.end()));
}

void InferenceWorker::run() {
    std::vector<ActiveSeq> active;
    std::vector<PrefillCtx> prefilling;
    const int32_t max_generating = static_cast<int32_t>(llama_n_seq_max(bridge_.ctx())) - 1;
    while (running_.load()) {
        try {
            std::vector<InferenceRequest> admitted;
            {
                std::unique_lock<std::mutex> lock(queue_mutex_);
                const bool idle = active.empty() && prefilling.empty();
                if (idle && queue_.empty()) {
                    queue_cv_.wait(lock, [this] { return !queue_.empty() || !running_.load(); });
                }
                if (!running_.load() && queue_.empty()) break;
                if (idle && !queue_.empty() && admit_window_ms_ > 0) {
                    const auto deadline = std::chrono::steady_clock::now() +
                        std::chrono::milliseconds(admit_window_ms_);
                    queue_cv_.wait_until(lock, deadline);
                }
                while (!queue_.empty() && static_cast<int32_t>(active.size() + prefilling.size() +
                                             admitted.size()) < max_generating) {
                    admitted.push_back(std::move(queue_.front()));
                    queue_.pop_front();
                }
            }

            std::vector<InferenceRequest> pending = std::move(admitted);
            std::vector<InferenceRequest> retry;
            size_t prepared_count = 0;
            if (!pending.empty()) {
                std::vector<InferenceRequest> deferred;
                std::vector<InferenceRequest> queued;
                auto ctxs = prepare_all(pending, &deferred, &queued);
                prepared_count = ctxs.size();
                prefilling.insert(prefilling.end(), std::make_move_iterator(ctxs.begin()),
                                  std::make_move_iterator(ctxs.end()));
                retry.insert(retry.end(), std::make_move_iterator(queued.begin()),
                              std::make_move_iterator(queued.end()));
                retry.insert(retry.end(), std::make_move_iterator(deferred.begin()),
                             std::make_move_iterator(deferred.end()));
            }
            defer_to_front(retry);

            generating_sequences_.store(static_cast<int32_t>(active.size()),
                                        std::memory_order_relaxed);
            fused_step(active, prefilling, prepared_count);
            generating_sequences_.store(static_cast<int32_t>(active.size()),
                                        std::memory_order_relaxed);
        } catch (const std::exception& e) {
            RF_WARN("worker", "Worker loop exception (non-fatal): %s", e.what());
        } catch (...) {
            RF_WARN("worker", "Worker loop unknown exception (non-fatal)");
        }
    }
    for (auto& seq : active) retire(seq, false);
    for (auto& ctx : prefilling) {
        mapper_.release_sequence(ctx.prep.seq_id, ctx.prep.leaf_node);
        tree_.release(ctx.prep.leaf_node);
    }
    generating_sequences_.store(0, std::memory_order_relaxed);
}


} // namespace radixforge
