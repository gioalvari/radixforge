#pragma once

#include "radixforge/config.h"
#include "radixforge/kv_mapper.h"
#include "radixforge/llama_bridge.h"
#include "radixforge/radix_tree.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace radixforge {

// A single chat turn (role + content)
struct ChatMessage {
    std::string role;
    std::string content;
};

// Per-request generation measurements compatible with llama-server's timings.
struct InferenceStats {
    int32_t prompt_n = 0;
    int32_t cache_n = 0;
    int32_t prompt_tokens = 0;
    int32_t predicted_n = 0;
    double prompt_ms = 0.0;
    double predicted_ms = 0.0;
};

// A generation request from an agent
struct InferenceRequest {
    std::string prompt;                 // raw prompt (used when messages is empty)
    std::vector<ChatMessage> messages;  // chat messages (preferred; native template applied)
    int32_t max_tokens = 4096;
    float temperature = 0.7f;
    float top_p = 0.9f;
    bool stream = true;

    // Callback for streaming tokens back (called from worker thread). stats is
    // non-null only for the final callback.
    std::function<void(const std::string& token, bool is_last,
                       const InferenceStats* stats)> on_token;
    // Returns true once the HTTP peer has disconnected or a synchronous request
    // has timed out. The worker polls it before admission and each decode step.
    std::function<bool()> cancelled;
};

// Internal prefill context.  It remains pending until every prompt token has
// been submitted successfully, allowing long prompts to be chunked across
// fused decode steps.
struct PrefillCtx {
    InferenceRequest req;
    KVMapper::PrepareResult prep;
    std::vector<llama_token> tokens;    // full prompt tokens (for full-cache-hit ref)
    int32_t delta_start = 0;            // number of delta tokens submitted
    InferenceStats stats;
};

// Internal state for one active sequence in the batched decode loop
struct ActiveSeq {
    InferenceRequest req;
    llama_seq_id seq_id = 0;
    RadixNode* leaf_node = nullptr;
    int32_t pos = 0;           // next decode position
    llama_token next_token = 0; // token queued for the next batch step
    int32_t remaining = 0;     // remaining tokens to generate (decremented each step)
    llama_sampler* sampler = nullptr;
    bool done = false;
    bool notify_on_retire = false;
    bool generation_started = false;
    InferenceStats stats;
    std::chrono::steady_clock::time_point generation_started_at;
};

// Single-threaded inference worker with continuous batching. New requests are
// admitted between one-token decode steps and completed sequences leave at once.
class InferenceWorker {
public:
    InferenceWorker(LlamaBridge& bridge, RadixTree& tree, KVMapper& mapper,
                    const Config& config);
    ~InferenceWorker();

    void start();
    void stop();
    void enqueue(InferenceRequest req);
    size_t pending_count() const;
    int32_t active_sequence_count() const;
    int32_t generating_sequence_count() const;

private:
    LlamaBridge& bridge_;
    RadixTree& tree_;
    KVMapper& mapper_;

    std::deque<InferenceRequest> queue_;
    mutable std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    std::atomic<bool> running_{false};
    std::thread worker_thread_;
    uint64_t request_count_ = 0;
    std::atomic<int32_t> generating_sequences_{0};
    int32_t admit_window_ms_ = 2;
    std::atomic<uint64_t> decode_steps_{0};
    std::atomic<uint64_t> prefill_tokens_{0};
    std::atomic<uint64_t> prefill_calls_{0};

    void run();
    // Step 1: tokenise + KV-prepare every request, build PrefillCtx list.
    //         Does NOT call llama_decode yet.
    std::vector<PrefillCtx> prepare_all(std::vector<InferenceRequest>& reqs,
                                         std::vector<InferenceRequest>* deferred,
                                         std::vector<InferenceRequest>* queued,
                                         int32_t* prompt_budget,
                                         bool allow_deferral = true);
    void defer_to_front(std::vector<InferenceRequest>& deferred);
    // Run one fused llama_decode call: queued generation tokens plus as many
    // prompt tokens as fit in the remaining n_batch capacity.
    void fused_step(std::vector<ActiveSeq>& seqs,
                    std::vector<PrefillCtx>& prefilling,
                    size_t newly_admitted);
    void retire(ActiveSeq& seq, bool notify_client);
    static bool is_cancelled(const InferenceRequest& req);

public:
    uint64_t decode_steps() const { return decode_steps_.load(std::memory_order_relaxed); }
    uint64_t prefill_tokens() const { return prefill_tokens_.load(std::memory_order_relaxed); }
    uint64_t prefill_calls() const { return prefill_calls_.load(std::memory_order_relaxed); }
};

// HTTP server using cpp-httplib (OpenAI-compatible /v1/chat/completions).
class Server {
public:
    Server(const Config& config, InferenceWorker& worker,
           KVMapper& mapper, RadixTree& tree);

    void start();  // blocking
    void stop();   // thread-safe; interrupts start() from another thread

private:
    Config config_;
    InferenceWorker& worker_;
    KVMapper& mapper_;
    RadixTree& tree_;
    std::atomic<bool> running_{false};
    std::function<void()> stop_fn_;  // set by start(), called by stop()
};

} // namespace radixforge
