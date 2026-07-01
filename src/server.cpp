#include "radixforge/server.h"
#include "radixforge/json_minimal.h"

#include "httplib.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <string>

namespace radixforge {

// --- InferenceWorker ---

InferenceWorker::InferenceWorker(LlamaBridge& bridge, RadixTree& tree, KVMapper& mapper)
    : bridge_(bridge), tree_(tree), mapper_(mapper) {}

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
        queue_.push(std::move(req));
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

// ── Phase 1: tokenise + KV-prepare every request ─────────────────────────────
// No llama_decode yet — just builds PrefillCtx for each request.
std::vector<PrefillCtx> InferenceWorker::prepare_all(
    std::vector<InferenceRequest>& reqs) {

    std::vector<PrefillCtx> ctxs;
    ctxs.reserve(reqs.size());

    for (auto& req : reqs) {
        // Resolve chat template
        std::string prompt = req.prompt;
        if (prompt.empty() && !req.messages.empty()) {
            std::vector<std::pair<std::string,std::string>> chat_msgs;
            chat_msgs.reserve(req.messages.size());
            for (const auto& m : req.messages)
                chat_msgs.push_back({m.role, m.content});
            prompt = bridge_.apply_chat_template(chat_msgs);
        }

        PrefillCtx ctx;
        ctx.req   = std::move(req);
        ctx.tokens = bridge_.tokenize(prompt, /*add_bos=*/true);

        // Guard: clamp max_tokens so the total never exceeds the context window.
        int32_t available = bridge_.n_ctx() - (int32_t)ctx.tokens.size() - 4;
        if (available <= 0) {
            fprintf(stderr, "[radixforge] Prompt too long (%zu tokens) for ctx %d — rejected\n",
                    ctx.tokens.size(), bridge_.n_ctx());
            if (ctx.req.on_token) ctx.req.on_token("[ERROR: prompt exceeds context window]", true);
            continue;
        }
        if (ctx.req.max_tokens > available) {
            fprintf(stderr, "[radixforge] max_tokens clamped %d → %d (ctx limit)\n",
                    ctx.req.max_tokens, available);
            ctx.req.max_tokens = available;
        }

        try {
            ctx.prep  = mapper_.prepare_sequence(ctx.tokens);
        } catch (const std::exception& e) {
            fprintf(stderr, "[radixforge] prepare_sequence failed: %s\n", e.what());
            if (ctx.req.on_token) ctx.req.on_token("[ERROR: resource exhausted]", true);
            continue;
        }

        ctx.delta_start = ctx.prep.cached_pos;  // index into tokens[] where delta begins

        fprintf(stderr, "[radixforge] Prefill: %zu tokens, cached: %d, delta: %zu\n",
                ctx.tokens.size(), ctx.prep.cached_pos, ctx.prep.delta.size());

        ctxs.push_back(std::move(ctx));
    }
    return ctxs;
}

// ── Phase 2: parallel prefill ────────────────────────────────────────────────
// Build one llama_batch containing ALL delta tokens from all requests,
// each tagged with its own seq_id and position. One GPU call covers everything.
// If the total token count exceeds n_batch, ctxs are split into sub-batches.
std::vector<ActiveSeq> InferenceWorker::prefill_batch(
    std::vector<PrefillCtx>& ctxs) {

    std::vector<ActiveSeq> all_active;
    if (ctxs.empty()) return all_active;

    const int32_t max_batch = bridge_.n_batch();

    // Partition ctxs into sub-batches, each fitting within max_batch tokens.
    size_t start = 0;
    while (start < ctxs.size()) {
        int32_t sub_total = 0;
        size_t end = start;
        while (end < ctxs.size()) {
            const auto& ctx = ctxs[end];
            int32_t tc = !ctx.prep.delta.empty()
                ? (int32_t)ctx.prep.delta.size()
                : (ctx.prep.cached_pos > 0 ? 1 : 0);
            if (tc == 0) { end++; continue; }
            if (end > start && sub_total + tc > max_batch) break;
            sub_total += tc;
            end++;
        }
        if (end == start) end = start + 1;  // at least one ctx per sub-batch

        auto sub = prefill_range(ctxs, start, end);
        for (auto& s : sub) all_active.push_back(std::move(s));
        start = end;
    }
    return all_active;
}

std::vector<ActiveSeq> InferenceWorker::prefill_range(
    std::vector<PrefillCtx>& ctxs, size_t from, size_t to) {

    std::vector<ActiveSeq> active;
    if (from >= to) return active;

    // --- Delta sequences: use prep.delta tokens ---
    // --- Full-cache-hit sequences: one re-decode token ---

    // Count total batch size for this sub-range
    int32_t total_tokens = 0;
    for (size_t ci = from; ci < to; ci++) {
        const auto& ctx = ctxs[ci];
        if (!ctx.prep.delta.empty()) {
            total_tokens += (int32_t)ctx.prep.delta.size();
        } else if (ctx.prep.cached_pos > 0) {
            total_tokens += 1;  // full-cache-hit: one re-decode token
        }
        // cached_pos == 0 && delta empty → empty prompt, skip
    }

    if (total_tokens == 0) return active;

    llama_batch batch = llama_batch_init(total_tokens, 0, 1);
    batch.n_tokens = total_tokens;

    int32_t batch_idx = 0;
    // track which batch index belongs to which ctx (for logit sampling)
    // indexed relative to `from`
    const size_t n_sub = to - from;
    std::vector<int32_t> ctx_last_logit_idx(n_sub, -1);

    for (size_t i = 0; i < n_sub; i++) {
        auto& ctx = ctxs[from + i];
        if (!ctx.prep.delta.empty()) {
            // Submit all delta tokens; only the LAST one needs logits
            for (int32_t di = 0; di < (int32_t)ctx.prep.delta.size(); di++) {
                batch.token[batch_idx]     = ctx.prep.delta[di];
                batch.pos[batch_idx]       = ctx.prep.cached_pos + di;
                batch.n_seq_id[batch_idx]  = 1;
                batch.seq_id[batch_idx][0] = ctx.prep.seq_id;
                batch.logits[batch_idx]    = (di == (int32_t)ctx.prep.delta.size() - 1) ? 1 : 0;
                batch_idx++;
            }
            ctx_last_logit_idx[i] = batch_idx - 1;
        } else if (ctx.prep.cached_pos > 0) {
            // Full cache hit — KV has [0..cached_pos-1]; decode last token at cached_pos
            // to get fresh logits.
            llama_token last_tok = ctx.tokens[ctx.prep.cached_pos - 1];
            batch.token[batch_idx]     = last_tok;
            batch.pos[batch_idx]       = ctx.prep.cached_pos;  // next free position
            batch.n_seq_id[batch_idx]  = 1;
            batch.seq_id[batch_idx][0] = ctx.prep.seq_id;
            batch.logits[batch_idx]    = 1;
            ctx_last_logit_idx[i] = batch_idx;
            batch_idx++;
        }
    }

    batch.n_tokens = batch_idx;  // adjust for any skipped ctxs

    bool ok = (llama_decode(bridge_.ctx(), batch) == 0);
    llama_batch_free(batch);

    if (!ok) {
        fprintf(stderr, "[radixforge] Parallel prefill decode failed\n");
        for (size_t i = 0; i < n_sub; i++) {
            auto& ctx = ctxs[from + i];
            if (ctx.req.on_token) ctx.req.on_token("[ERROR: prefill failed]", true);
            mapper_.release_sequence(ctx.prep.seq_id, ctx.prep.leaf_node);
            tree_.release(ctx.prep.leaf_node);
        }
        return active;
    }

    // Sample first token for each ctx using its batch logit index
    for (size_t i = 0; i < n_sub; i++) {
        auto& ctx = ctxs[from + i];
        int32_t logit_idx = ctx_last_logit_idx[i];
        if (logit_idx < 0) continue;  // empty prompt, skip

        llama_sampler* sampler = bridge_.create_sampler(ctx.req.temperature, ctx.req.top_p);
        llama_token first_tok  = llama_sampler_sample(sampler, bridge_.ctx(), logit_idx);

        if (llama_vocab_is_eog(bridge_.vocab(), first_tok)) {
            if (ctx.req.on_token) ctx.req.on_token("", true);
            LlamaBridge::free_sampler(sampler);
            mapper_.release_sequence(ctx.prep.seq_id, ctx.prep.leaf_node);
            tree_.release(ctx.prep.leaf_node);
            continue;
        }

        char buf[512];
        int32_t nc = llama_token_to_piece(bridge_.vocab(), first_tok, buf, sizeof(buf), 0, true);
        std::string piece(buf, nc > 0 ? nc : 0);
        bool first_is_last = (ctx.req.max_tokens <= 1);
        if (ctx.req.on_token) ctx.req.on_token(piece, first_is_last);

        if (first_is_last) {
            LlamaBridge::free_sampler(sampler);
            mapper_.release_sequence(ctx.prep.seq_id, ctx.prep.leaf_node);
            tree_.release(ctx.prep.leaf_node);
            continue;
        }

        // Compute position after prefill
        int32_t decode_end_pos = ctx.prep.delta.empty()
            ? ctx.prep.cached_pos + 1   // full-cache-hit: decoded at cached_pos
            : ctx.prep.cached_pos + (int32_t)ctx.prep.delta.size();

        ActiveSeq seq;
        seq.seq_id     = ctx.prep.seq_id;
        seq.leaf_node  = ctx.prep.leaf_node;
        seq.pos        = decode_end_pos;
        seq.next_token = first_tok;
        seq.remaining  = ctx.req.max_tokens - 1;
        seq.sampler    = sampler;
        seq.req        = std::move(ctx.req);
        active.push_back(std::move(seq));
    }

    return active;
}

// ── Phase 3: batched autoregressive loop ─────────────────────────────────────
void InferenceWorker::run_batch(std::vector<ActiveSeq>& seqs) {
    while (!seqs.empty()) {
        int n = (int)seqs.size();
        llama_batch batch = llama_batch_init(n, 0, 1);
        batch.n_tokens = n;
        for (int i = 0; i < n; i++) {
            batch.token[i]     = seqs[i].next_token;
            batch.pos[i]       = seqs[i].pos;
            batch.n_seq_id[i]  = 1;
            batch.seq_id[i][0] = seqs[i].seq_id;
            batch.logits[i]    = 1;
        }

        bool ok = (llama_decode(bridge_.ctx(), batch) == 0);
        llama_batch_free(batch);

        if (!ok) {
            fprintf(stderr, "[radixforge] Batch decode failed (%d seqs)\n", n);
            for (auto& s : seqs) if (s.req.on_token) s.req.on_token("", true);
            for (auto& s : seqs) {
                LlamaBridge::free_sampler(s.sampler);
                mapper_.release_sequence(s.seq_id, s.leaf_node);
                tree_.release(s.leaf_node);
            }
            seqs.clear();
            return;
        }

        for (int i = 0; i < n; i++) {
            ActiveSeq& seq = seqs[i];
            llama_token new_tok = llama_sampler_sample(seq.sampler, bridge_.ctx(), i);

            if (llama_vocab_is_eog(bridge_.vocab(), new_tok)) {
                if (seq.req.on_token) seq.req.on_token("", true);
                seq.done = true;
                continue;
            }

            char buf[512];
            int32_t nc = llama_token_to_piece(bridge_.vocab(), new_tok, buf, sizeof(buf), 0, true);
            std::string piece(buf, nc > 0 ? nc : 0);

            seq.remaining--;
            bool last = (seq.remaining == 0);
            if (seq.req.on_token) seq.req.on_token(piece, last);

            if (last) {
                seq.done = true;
            } else {
                seq.next_token = new_tok;
                seq.pos++;
            }
        }

        auto end_it = std::remove_if(seqs.begin(), seqs.end(), [&](ActiveSeq& s) {
            if (s.done) {
                LlamaBridge::free_sampler(s.sampler);
                mapper_.release_sequence(s.seq_id, s.leaf_node);
                tree_.release(s.leaf_node);
                return true;
            }
            return false;
        });
        seqs.erase(end_it, seqs.end());
    }
}

void InferenceWorker::run() {
    while (running_.load()) {
      try {
        std::vector<InferenceRequest> reqs;
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            queue_cv_.wait(lock, [this] {
                return !queue_.empty() || !running_.load();
            });
            if (!running_.load() && queue_.empty()) break;
            while (!queue_.empty()) {
                reqs.push_back(std::move(queue_.front()));
                queue_.pop();
            }
        }

        // Phase 1: tokenise + KV-prepare (no GPU)
        auto ctxs = prepare_all(reqs);

        // Phase 2: one GPU prefill call for all deltas
        if (!ctxs.empty()) {
            fprintf(stderr, "[radixforge] Parallel prefill: %zu sequences\n", ctxs.size());
            auto active = prefill_batch(ctxs);

            // Phase 3: batched autoregressive generation
            if (!active.empty()) {
                fprintf(stderr, "[radixforge] Batch generation: %zu sequences\n", active.size());
                run_batch(active);
            }
        }

        request_count_ += reqs.size();
        if (request_count_ % 64 < reqs.size()) mapper_.defrag();

      } catch (const std::exception& e) {
        fprintf(stderr, "[radixforge] Worker loop exception (non-fatal): %s\n", e.what());
      } catch (...) {
        fprintf(stderr, "[radixforge] Worker loop unknown exception (non-fatal)\n");
      }
    }
}

// --- Server (cpp-httplib) ---

Server::Server(const Config& config, InferenceWorker& worker)
    : config_(config), worker_(worker) {}

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

    // Health check
    svr.Get("/health", [this](const httplib::Request&, httplib::Response& res) {
        size_t pending = worker_.pending_count();
        int32_t active_seqs = worker_.active_sequence_count();
        res.set_header("Access-Control-Allow-Origin", "*");
        res.set_content(
            "{\"status\":\"ok\",\"active_sequences\":" + std::to_string(active_seqs) +
            ",\"pending_requests\":" + std::to_string(pending) + "}",
            "application/json");
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
                bool done = false;
            };
            auto state = std::make_shared<StreamState>();

            res.set_chunked_content_provider("text/event-stream",
                [this, req = std::move(req), state](size_t /*offset*/,
                                                     httplib::DataSink& sink) mutable -> bool {
                    // Wire on_token to write directly to sink (only valid inside this lambda).
                    req.on_token = [state, &sink](const std::string& token, bool is_last) {
                        if (state->cancelled.load(std::memory_order_relaxed)) return;
                        std::lock_guard<std::mutex> lk(state->mtx);
                        if (!sink.is_writable()) {
                            state->cancelled.store(true, std::memory_order_relaxed);
                            state->done = true;
                            state->cv.notify_one();
                            return;
                        }
                        std::string chunk =
                            "data: {\"choices\":[{\"delta\":{\"content\":\""
                            + json_escape_s(token) + "\"},\"finish_reason\":"
                            + (is_last ? "\"stop\"" : "null") + "}]}\n\n";
                        sink.write(chunk.c_str(), chunk.size());
                        if (is_last) {
                            sink.write("data: [DONE]\n\n", 14);
                            state->done = true;
                            state->cv.notify_one();
                        }
                    };

                    worker_.enqueue(std::move(req));

                    // Keepalive loop — send empty delta every 2s while waiting
                    std::unique_lock<std::mutex> lk(state->mtx);
                    while (!state->cv.wait_for(lk, std::chrono::seconds(2),
                                               [&]{ return state->done; })) {
                        if (!sink.is_writable()) {
                            state->cancelled.store(true, std::memory_order_relaxed);
                            return false;
                        }
                        static const std::string ka =
                            "data: {\"choices\":[{\"delta\":{},\"finish_reason\":null}]}\n\n";
                        sink.write(ka.c_str(), ka.size());
                    }
                    return true;  // close stream
                });
        } else {
            // Non-streaming: state lives in a shared_ptr — safe even if this
            // handler returns (timeout) before the worker calls on_token.
            struct NonStreamState {
                std::mutex mtx;
                std::condition_variable cv;
                std::string full_response;
                std::atomic<bool> cancelled{false};
                bool done = false;
            };
            auto state = std::make_shared<NonStreamState>();

            req.stream = false;
            req.on_token = [state](const std::string& token, bool is_last) {
                if (state->cancelled.load(std::memory_order_relaxed)) return;
                std::lock_guard<std::mutex> lk(state->mtx);
                state->full_response += token;
                if (is_last) { state->done = true; state->cv.notify_one(); }
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
                "\"},\"finish_reason\":\"stop\"}]}";
            res.set_content(json_resp, "application/json");
        }
    };

    svr.Post("/v1/chat/completions", handle_chat);
    svr.Post("/v1/completions",       handle_chat);

    // stop_fn_ captures svr_p by value — shared_ptr keeps server alive
    // even after start() returns, so stop() called from signal handler is safe.
    stop_fn_ = [svr_p]() { svr_p->stop(); };

    fprintf(stderr, "[radixforge] HTTP server (httplib) listening on %s:%d\n",
            config_.host.c_str(), config_.port);

    svr.listen(config_.host.c_str(), config_.port);
    running_.store(false);
}

void Server::stop() {
    if (stop_fn_) stop_fn_();
    running_.store(false);
}

} // namespace radixforge
