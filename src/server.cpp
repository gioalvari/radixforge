#include "radixforge/server.h"
#include "radixforge/json_minimal.h"
#include "radixforge/logger.h"

#include "httplib.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <iterator>
#include <string>

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
    std::vector<InferenceRequest>* queued, int32_t* prompt_budget,
    bool allow_deferral) {

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
            if (req.on_token) {
                req.on_token("[ERROR: prompt exceeds context window]", true, nullptr);
            }
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
        if (allow_deferral && pending_advantage > 0) {
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
            if (ctx.req.on_token) {
                ctx.req.on_token("[ERROR: resource exhausted]", true, nullptr);
            }
            continue;
        }

        const int32_t prompt_tokens = !ctx.prep.delta.empty()
            ? static_cast<int32_t>(ctx.prep.delta.size())
            : (ctx.prep.cached_pos > 0 ? 1 : 0);
        if (prompt_budget && !ctxs.empty() &&
            prompt_tokens > *prompt_budget) {
            mapper_.release_sequence(ctx.prep.seq_id, ctx.prep.leaf_node);
            tree_.release(ctx.prep.leaf_node);
            if (queued) {
                queued->push_back(std::move(ctx.req));
                for (++req_index; req_index < reqs.size(); ++req_index) {
                    if (!is_cancelled(reqs[req_index])) {
                        queued->push_back(std::move(reqs[req_index]));
                    }
                }
            }
            break;
        }
        if (prompt_budget) *prompt_budget -= std::min(prompt_tokens, *prompt_budget);
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
    if (notify_client && seq.req.on_token && !is_cancelled(seq.req)) {
        const auto finished = std::chrono::steady_clock::now();
        seq.stats.predicted_ms = std::chrono::duration<double, std::milli>(
            finished - seq.generation_started_at).count();
        seq.req.on_token("", true, &seq.stats);
    }
    LlamaBridge::free_sampler(seq.sampler);
    mapper_.release_sequence(seq.seq_id, seq.leaf_node);
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
        const int32_t total = !ctx.prep.delta.empty()
            ? static_cast<int32_t>(ctx.prep.delta.size())
            : (ctx.prep.cached_pos > 0 ? 1 : 0);
        const int32_t remaining = total - ctx.delta_start;
        if (remaining <= 0) continue;
        const int32_t take = std::min(remaining, capacity - prompt_tokens);
        const bool completes = take == remaining;
        const int32_t start = ctx.delta_start;
        for (int32_t offset = 0; offset < take; ++offset) {
            const int32_t prompt_index = start + offset;
            batch.token[batch_idx] = !ctx.prep.delta.empty()
                ? ctx.prep.delta[prompt_index]
                : ctx.tokens[ctx.prep.cached_pos - 1];
            batch.pos[batch_idx] = !ctx.prep.delta.empty()
                ? ctx.prep.cached_pos + prompt_index
                : ctx.prep.cached_pos;
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

    if (decode_status == 1) {
        RF_WARN("worker", "Step stalled by full KV cache; keeping sequences queued");
        return;
    }
    if (decode_status < 0) {
        RF_ERROR("worker", "Fused batch decode failed (%d seqs, %d prompt tokens, status %d)",
                 decoding, prompt_tokens, decode_status);
        for (auto& seq : seqs) retire(seq, true);
        seqs.clear();
        for (auto& ctx : prefilling) {
            if (ctx.req.on_token && !is_cancelled(ctx.req)) {
                ctx.req.on_token("[ERROR: prefill failed]", true, nullptr);
            }
            mapper_.release_sequence(ctx.prep.seq_id, ctx.prep.leaf_node);
            tree_.release(ctx.prep.leaf_node);
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
        const int32_t total = !ctx.prep.delta.empty()
            ? static_cast<int32_t>(ctx.prep.delta.size())
            : (ctx.prep.cached_pos > 0 ? 1 : 0);
        if (ctx.delta_start == total) {
            mapper_.mark_sequence_ready(ctx.prep.seq_id, ctx.prep.leaf_node);
            prompt_logits[submission.ctx_index] = submission.logit_index;
        }
    }

    const auto now = std::chrono::steady_clock::now();
    for (int i = 0; i < decoding; ++i) {
        ActiveSeq& seq = seqs[i];
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
        const int32_t end_pos = ctx.prep.delta.empty()
            ? ctx.prep.cached_pos + 1
            : ctx.prep.cached_pos + static_cast<int32_t>(ctx.prep.delta.size());
        ActiveSeq seq;
        seq.seq_id = ctx.prep.seq_id;
        seq.leaf_node = ctx.prep.leaf_node;
        seq.pos = end_pos;
        seq.next_token = first_tok;
        seq.remaining = ctx.req.max_tokens - 1;
        seq.sampler = sampler;
        seq.stats = ctx.stats;
        seq.req = std::move(ctx.req);
        seq.generation_started = true;
        seq.generation_started_at = now;
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
                auto ctxs = prepare_all(pending, &deferred, &queued, nullptr);
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
            ++request_count_;
            if (request_count_ % 64 == 0) mapper_.defrag();
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

// --- Server (cpp-httplib) ---

Server::Server(const Config& config, InferenceWorker& worker,
               KVMapper& mapper, RadixTree& tree)
    : config_(config), worker_(worker), mapper_(mapper), tree_(tree) {}

// Returns true if the request passes admin auth (or auth is disabled).
static bool check_admin_auth(const Config& config,
                              const httplib::Request& req,
                              httplib::Response& res) {
    if (config.admin_token.empty()) return true;
    const std::string expected = "Bearer " + config.admin_token;
    if (req.get_header_value("Authorization") == expected) return true;
    res.status = 401;
    res.set_content("{\"error\":\"Unauthorized — set Authorization: Bearer <admin-token>\"}",
                    "application/json");
    return false;
}

static std::string timings_json(const InferenceStats& stats) {
    return "{\"prompt_n\":" + std::to_string(stats.prompt_n) +
           ",\"cache_n\":" + std::to_string(stats.cache_n) +
           ",\"prompt_ms\":" + std::to_string(stats.prompt_ms) +
           ",\"predicted_n\":" + std::to_string(stats.predicted_n) +
           ",\"predicted_ms\":" + std::to_string(stats.predicted_ms) + "}";
}

static std::string usage_json(const InferenceStats& stats) {
    return "{\"prompt_tokens\":" + std::to_string(stats.prompt_tokens) +
           ",\"completion_tokens\":" + std::to_string(stats.predicted_n) +
           ",\"total_tokens\":" +
           std::to_string(stats.prompt_tokens + stats.predicted_n) + "}";
}

static void parse_request_body(const httplib::Request& req_http,
                               InferenceRequest& req,
                               const Config& config) {
    std::string body_str = req_http.body;
    json::JsonValue body = json::parse(body_str);

    if (body.contains("messages") && body["messages"].type() == json::Type::Array) {
        for (size_t i = 0; i < body["messages"].size(); i++) {
            const auto& msg = body["messages"][i];
            ChatMessage cm;
            cm.role    = msg.contains("role")    ? msg["role"].as_string()    : "user";
            cm.content = msg.contains("content") ? msg["content"].as_string() : "";
            req.messages.push_back(std::move(cm));
        }
    } else if (body.contains("prompt") && body["prompt"].type() == json::Type::String) {
        req.prompt = body["prompt"].as_string();
    }

    if (body.contains("max_tokens") && body["max_tokens"].type() == json::Type::Number)
        req.max_tokens = (int32_t)body["max_tokens"].as_number();
    else
        req.max_tokens = config.max_tokens;

    if (body.contains("temperature") && body["temperature"].type() == json::Type::Number)
        req.temperature = (float)body["temperature"].as_number();
    else
        req.temperature = config.temperature;

    if (body.contains("top_p") && body["top_p"].type() == json::Type::Number)
        req.top_p = (float)body["top_p"].as_number();
    else
        req.top_p = config.top_p;

    req.stream = body.contains("stream") &&
                 body["stream"].type() == json::Type::Bool &&
                 body["stream"].as_bool();
}

static std::string json_escape_s(const std::string& s) {
    std::string out; out.reserve(s.size() + 16);
    for (unsigned char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (c < 0x20) { char b[8]; snprintf(b,8,"\\u%04x",c); out+=b; }
                else out += (char)c;
        }
    }
    return out;
}

void Server::start() {
    running_.store(true);

    // Heap-allocate the httplib server so stop_fn_ can safely capture it by
    // shared_ptr value. Calling stop() after listen() returns is idempotent.
    auto svr_p = std::make_shared<httplib::Server>();
    httplib::Server& svr = *svr_p;  // reference alias — all routes use svr.xxx unchanged
    svr.set_keep_alive_max_count(100);
    svr.set_read_timeout(30);
    svr.set_write_timeout(120);

    // CORS preflight
    svr.Options(".*", [](const httplib::Request&, httplib::Response& res) {
        res.set_header("Access-Control-Allow-Origin", "*");
        res.set_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS");
        res.set_header("Access-Control-Allow-Headers", "Content-Type, Authorization");
        res.status = 204;
    });

    // Health check — includes model name and pool status
    svr.Get("/health", [this](const httplib::Request&, httplib::Response& res) {
        size_t pending = worker_.pending_count();
        int32_t active_seqs = worker_.active_sequence_count();
        res.set_header("Access-Control-Allow-Origin", "*");
        const CacheMetrics& m = mapper_.metrics();
        uint64_t total = m.total_requests.load(std::memory_order_relaxed);
        uint64_t hits  = m.hits.load(std::memory_order_relaxed);
        double hit_rate = total > 0 ? static_cast<double>(hits) / total : 0.0;
        char buf[512];
        snprintf(buf, sizeof(buf),
            "{\"status\":\"ok\",\"model\":\"%s\","
            "\"active_sequences\":%d,\"pending_requests\":%zu,"
            "\"cache_hit_rate\":%.4f}",
            config_.model_path.substr(config_.model_path.rfind('/') + 1).c_str(),
            active_seqs, pending, hit_rate);
        res.set_content(buf, "application/json");
    });

    // /admin/stats — cache metrics and worker state (auth required if admin_token set)
    svr.Get("/admin/stats", [this](const httplib::Request& req, httplib::Response& res) {
        if (!check_admin_auth(config_, req, res)) return;
        const CacheMetrics& m = mapper_.metrics();
        uint64_t hits  = m.hits.load(std::memory_order_relaxed);
        uint64_t total = m.total_requests.load(std::memory_order_relaxed);
        double hit_rate = total > 0 ? static_cast<double>(hits) / total : 0.0;
        char buf[768];
        snprintf(buf, sizeof(buf),
            "{\"active_sequences\":%d,\"generating_sequences\":%d,\"pending_requests\":%zu,"
            "\"decode_steps\":%llu,\"prefill_tokens\":%llu,\"prefill_calls\":%llu,"
            "\"cache\":{\"hits\":%llu,\"misses\":%llu,\"evictions\":%llu,"
            "\"total_requests\":%llu,\"hit_rate\":%.4f}}",
            worker_.active_sequence_count(), worker_.generating_sequence_count(),
            worker_.pending_count(),
            (unsigned long long)worker_.decode_steps(),
            (unsigned long long)worker_.prefill_tokens(),
            (unsigned long long)worker_.prefill_calls(),
            (unsigned long long)hits,
            (unsigned long long)m.misses.load(std::memory_order_relaxed),
            (unsigned long long)m.evictions.load(std::memory_order_relaxed),
            (unsigned long long)total,
            hit_rate);
        res.set_header("Access-Control-Allow-Origin", "*");
        res.set_content(buf, "application/json");
    });

    // /admin/tree/dump — radix tree structure as JSON (auth required if admin_token set)
    svr.Get("/admin/tree/dump", [this](const httplib::Request& req, httplib::Response& res) {
        if (!check_admin_auth(config_, req, res)) return;
        res.set_header("Access-Control-Allow-Origin", "*");
        res.set_content(tree_.to_json(), "application/json");
    });

    // /admin/seq/:id — force-evict a specific physical sequence (auth required)
    svr.Delete(R"(/admin/seq/(\d+))", [this](const httplib::Request& req, httplib::Response& res) {
        if (!check_admin_auth(config_, req, res)) return;
        llama_seq_id target;
        try { target = static_cast<llama_seq_id>(std::stoi(req.matches[1])); }
        catch (...) {
            res.status = 400;
            res.set_content("{\"error\":\"invalid seq_id\"}", "application/json");
            return;
        }
        bool ok = mapper_.force_evict_seq(target);
        res.set_header("Access-Control-Allow-Origin", "*");
        if (ok) {
            res.set_content("{\"status\":\"evicted\",\"seq_id\":" +
                            std::to_string(target) + "}", "application/json");
        } else {
            res.status = 404;
            res.set_content("{\"error\":\"seq_id not found or already free\"}", "application/json");
        }
    });

    // Chat completions (non-streaming)
    auto handle_chat = [this](const httplib::Request& req_http,
                               httplib::Response& res) {
        InferenceRequest req;
        try {
            parse_request_body(req_http, req, config_);
        } catch (...) {
            res.status = 400;
            res.set_content("{\"error\":\"invalid request body\"}", "application/json");
            return;
        }

        if (req.prompt.empty() && req.messages.empty()) {
            res.status = 400;
            res.set_content("{\"error\":\"no prompt or messages\"}", "application/json");
            return;
        }

        res.set_header("Access-Control-Allow-Origin", "*");

        if (req.stream) {
            // SSE streaming: state lives in a shared_ptr so the worker can safely
            // call on_token even after the client disconnects (sink becomes invalid).
            struct StreamState {
                std::mutex mtx;
                std::condition_variable cv;
                std::atomic<bool> cancelled{false};
                bool enqueued = false;
                bool done = false;
            };
            auto state = std::make_shared<StreamState>();

            res.set_chunked_content_provider("text/event-stream",
                [this, req = std::move(req), state](size_t /*offset*/,
                                                      httplib::DataSink& sink) mutable -> bool {
                    bool enqueue_request = false;
                    {
                        std::lock_guard<std::mutex> lk(state->mtx);
                        if (!state->enqueued) {
                            // The sink is valid until this provider returns. The
                            // mutex synchronizes disconnect with worker writes.
                            req.on_token = [state, &sink](const std::string& token,
                                                           bool is_last,
                                                           const InferenceStats* stats) {
                                std::lock_guard<std::mutex> token_lk(state->mtx);
                                if (state->cancelled.load(std::memory_order_relaxed)) return;
                                if (!sink.is_writable()) {
                                    state->cancelled.store(true, std::memory_order_relaxed);
                                    state->done = true;
                                    state->cv.notify_one();
                                    return;
                                }
                                std::string chunk =
                                    "data: {\"choices\":[{\"delta\":{\"content\":\""
                                    + json_escape_s(token) + "\"},\"finish_reason\":"
                                    + (is_last ? "\"stop\"" : "null")
                                    + (is_last && stats
                                           ? "}],\"timings\":" + timings_json(*stats) +
                                                 ",\"usage\":" + usage_json(*stats)
                                           : "")
                                    + (is_last && stats ? "}" : "}]}" ) + "\n\n";
                                if (!sink.write(chunk.c_str(), chunk.size())) {
                                    state->cancelled.store(true, std::memory_order_relaxed);
                                    state->done = true;
                                    state->cv.notify_one();
                                    return;
                                }
                                if (is_last) {
                                    if (!sink.write("data: [DONE]\n\n", 14)) {
                                        state->cancelled.store(true, std::memory_order_relaxed);
                                    }
                                    state->done = true;
                                    state->cv.notify_one();
                                }
                            };
                            req.cancelled = [state] {
                                return state->cancelled.load(std::memory_order_relaxed);
                            };
                            state->enqueued = true;
                            enqueue_request = true;
                        }
                    }

                    if (enqueue_request) worker_.enqueue(std::move(req));

                    // A completed provider must call done(); returning true alone
                    // asks cpp-httplib to invoke this lambda again.
                    std::unique_lock<std::mutex> lk(state->mtx);
                    while (!state->cv.wait_for(lk, std::chrono::seconds(2),
                                               [&]{ return state->done; })) {
                        if (!sink.is_writable()) {
                            state->cancelled.store(true, std::memory_order_relaxed);
                            return false;
                        }
                        static const std::string ka =
                            "data: {\"choices\":[{\"delta\":{},\"finish_reason\":null}]}\n\n";
                        if (!sink.write(ka.c_str(), ka.size())) {
                            state->cancelled.store(true, std::memory_order_relaxed);
                            return false;
                        }
                    }
                    if (state->cancelled.load(std::memory_order_relaxed)) {
                        lk.unlock();
                        return false;
                    }
                    lk.unlock();
                    sink.done();
                    return true;
                });
        } else {
            // Non-streaming: state lives in a shared_ptr — safe even if this
            // handler returns (timeout) before the worker calls on_token.
            struct NonStreamState {
                std::mutex mtx;
                std::condition_variable cv;
                std::string full_response;
                InferenceStats stats;
                std::atomic<bool> cancelled{false};
                bool done = false;
            };
            auto state = std::make_shared<NonStreamState>();

            req.stream = false;
            req.on_token = [state](const std::string& token, bool is_last,
                                   const InferenceStats* stats) {
                if (state->cancelled.load(std::memory_order_relaxed)) return;
                std::lock_guard<std::mutex> lk(state->mtx);
                state->full_response += token;
                if (is_last) {
                    if (stats) state->stats = *stats;
                    state->done = true;
                    state->cv.notify_one();
                }
            };
            req.cancelled = [state] {
                return state->cancelled.load(std::memory_order_relaxed);
            };

            worker_.enqueue(std::move(req));

            std::unique_lock<std::mutex> lk(state->mtx);
            bool completed = state->cv.wait_for(lk, std::chrono::seconds(60),
                                                [&]{ return state->done; });
            if (!completed) {
                // Mark cancelled BEFORE returning so the worker's on_token is a no-op.
                state->cancelled.store(true, std::memory_order_relaxed);
                res.status = 504;
                res.set_content("{\"error\":\"inference timeout\"}", "application/json");
                return;
            }

            std::string json_resp =
                "{\"choices\":[{\"message\":{\"role\":\"assistant\","
                "\"content\":\"" + json_escape_s(state->full_response) +
                "\"},\"finish_reason\":\"stop\"}],\"timings\":" +
                timings_json(state->stats) + ",\"usage\":" +
                usage_json(state->stats) + "}";
            res.set_content(json_resp, "application/json");
        }
    };

    svr.Post("/v1/chat/completions", handle_chat);
    svr.Post("/v1/completions",       handle_chat);

    // stop_fn_ captures svr_p by value — shared_ptr keeps server alive
    // even after start() returns, so stop() called from signal handler is safe.
    stop_fn_ = [svr_p]() { svr_p->stop(); };

    RF_INFO("server", "HTTP server listening on %s:%d",
            config_.host.c_str(), config_.port);

    svr.listen(config_.host.c_str(), config_.port);
    running_.store(false);
}

void Server::stop() {
    if (stop_fn_) stop_fn_();
    running_.store(false);
}

} // namespace radixforge
