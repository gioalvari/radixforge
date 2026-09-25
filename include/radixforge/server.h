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
};

// Internal prefill context — collected during the parallel prefill phase
struct PrefillCtx {
    InferenceRequest req;
    KVMapper::PrepareResult prep;
    std::vector<llama_token> tokens;    // full prompt tokens (for full-cache-hit ref)
    int32_t delta_start = 0;            // index into tokens[] where delta begins
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
    InferenceStats stats;
    std::chrono::steady_clock::time_point generation_started;
};

// Single-threaded inference worker that processes requests from a queue.
// Prefill is now parallelised: all pending requests have their delta tokens
// batched into a single llama_decode call before entering the autoregressive loop.
class InferenceWorker {
public:
    InferenceWorker(LlamaBridge& bridge, RadixTree& tree, KVMapper& mapper);
    ~InferenceWorker();

    void start();
    void stop();
    void enqueue(InferenceRequest req);
    size_t pending_count() const;
    int32_t active_sequence_count() const;

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

    void run();
    // Step 1: tokenise + KV-prepare every request, build PrefillCtx list.
    //         Does NOT call llama_decode yet.
    std::vector<PrefillCtx> prepare_all(std::vector<InferenceRequest>& reqs,
                                        std::vector<InferenceRequest>* deferred,
                                        bool allow_deferral = true);
    void defer_to_front(std::vector<InferenceRequest>& deferred);
    // Step 2: submit all delta tokens in one llama_decode batch.
    //         Returns active seqs ready for the autoregressive loop.
    std::vector<ActiveSeq> prefill_batch(std::vector<PrefillCtx>& ctxs);
    // Internal: run one sub-batch [from, to) — called by prefill_batch for chunking.
    std::vector<ActiveSeq> prefill_range(std::vector<PrefillCtx>& ctxs, size_t from, size_t to);
    // Step 3: batched autoregressive decode loop until all seqs are done.
    void run_batch(std::vector<ActiveSeq>& seqs);
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
