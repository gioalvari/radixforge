#include "radixforge/server.h"
#include "radixforge/logger.h"
#include "radixforge/openai_api.h"

#include "httplib.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <memory>
#include <mutex>
#include <string>

namespace radixforge {

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

    // Chat completions (non-streaming)
    auto handle_completion = [this](bool is_chat, const httplib::Request& req_http,
                                    httplib::Response& res) {
        InferenceRequest req;
        try {
            openai_api::parse_request_body(req_http.body, req, config_);
        } catch (const std::exception& e) {
            res.status = 400;
            res.set_content(openai_api::error_json(e.what(), "invalid_request_error"), "application/json");
            return;
        }

        if (req.prompt.empty() && req.messages.empty()) {
            res.status = 400;
            res.set_content(openai_api::error_json("no prompt or messages", "invalid_request_error"),
                            "application/json");
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
                std::string incomplete_utf8;
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
                                std::string content = state->incomplete_utf8 + token;
                                const size_t complete_length = openai_api::complete_utf8_prefix_length(content);
                                state->incomplete_utf8 = content.substr(complete_length);
                                content.resize(complete_length);
                                if (!state->incomplete_utf8.empty() && is_last) {
                                    content += state->incomplete_utf8;
                                    state->incomplete_utf8.clear();
                                }
                                std::string chunk =
                                    "data: {\"choices\":[{\"delta\":{\"content\":\""
                                    + openai_api::json_escape(content) + "\"},\"finish_reason\":"
                                    + (is_last ? "\"stop\"" : "null")
                                    + (is_last && stats
                                           ? "}],\"timings\":" + openai_api::timings_json(*stats) +
                                                 ",\"usage\":" + openai_api::usage_json(*stats)
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
                            req.on_error = [state, &sink](const std::string& message, int status) {
                                std::lock_guard<std::mutex> token_lk(state->mtx);
                                if (state->cancelled.load(std::memory_order_relaxed)) return;
                                const std::string event = "data: " + openai_api::error_json(
                                    message, status == 400 ? "invalid_request_error" : "server_error") +
                                    "\n\n";
                                if (!sink.is_writable() || !sink.write(event.c_str(), event.size()) ||
                                    !sink.write("data: [DONE]\n\n", 14)) {
                                    state->cancelled.store(true, std::memory_order_relaxed);
                                }
                                state->done = true;
                                state->cv.notify_one();
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
                int error_status = 0;
                std::string error_message;
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
            req.on_error = [state](const std::string& message, int status) {
                if (state->cancelled.load(std::memory_order_relaxed)) return;
                std::lock_guard<std::mutex> lk(state->mtx);
                state->error_status = status;
                state->error_message = message;
                state->done = true;
                state->cv.notify_one();
            };

            worker_.enqueue(std::move(req));

            std::unique_lock<std::mutex> lk(state->mtx);
            bool completed = state->cv.wait_for(
                lk, std::chrono::seconds(config_.request_timeout_seconds),
                                                 [&]{ return state->done; });
            if (!completed) {
                // Mark cancelled BEFORE returning so the worker's on_token is a no-op.
                state->cancelled.store(true, std::memory_order_relaxed);
                res.status = 504;
                res.set_content(openai_api::error_json("inference timeout", "timeout_error"),
                                "application/json");
                return;
            }

            if (state->error_status != 0) {
                res.status = state->error_status;
                res.set_content(openai_api::error_json(state->error_message,
                                           state->error_status == 400
                                               ? "invalid_request_error"
                                               : "server_error"),
                                "application/json");
                return;
            }

            const std::string json_resp = openai_api::completion_json(
                state->full_response, state->stats, is_chat);
            res.set_content(json_resp, "application/json");
        }
    };

    svr.Post("/v1/chat/completions", [handle_completion](const httplib::Request& req,
                                                           httplib::Response& res) {
        handle_completion(true, req, res);
    });
    svr.Post("/v1/completions", [handle_completion](const httplib::Request& req,
                                                      httplib::Response& res) {
        handle_completion(false, req, res);
    });

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
