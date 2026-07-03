#include "radixforge/config.h"
#include "radixforge/kv_mapper.h"
#include "radixforge/llama_bridge.h"
#include "radixforge/logger.h"
#include "radixforge/radix_tree.h"
#include "radixforge/server.h"

#include <cstdio>
#include <cstdlib>
#include <csignal>
#include <string>

static radixforge::Server* g_server = nullptr;

static void signal_handler(int) {
    if (g_server) g_server->stop();
}

static void print_usage(const char* prog) {
    fprintf(stderr,
        "Usage: %s -m <model.gguf> [options]\n"
        "Options:\n"
        "  -m, --model <path>         Path to GGUF model (required)\n"
        "  -c, --ctx-size <n>         Context size (default: 32768)\n"
        "  -b, --batch-size <n>       Max decode batch size (default: 2048)\n"
        "  -ngl, --n-gpu-layers       GPU layers to offload (default: 99)\n"
        "  --host <addr>              Listen address (default: 127.0.0.1)\n"
        "  --port <n>                 Listen port (default: 8400)\n"
        "  --max-seq <n>              Max concurrent sequences (default: 32)\n"
        "  --admin-token <token>      Bearer token for /admin/* endpoints (default: none)\n"
        "  --log-level <level>        Log level: debug|info|warn|error (default: info)\n"
        "\n", prog);
}

int main(int argc, char** argv) {
    radixforge::Config config;

    // Parse CLI arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if ((arg == "-m" || arg == "--model") && i + 1 < argc) {
            config.model_path = argv[++i];
        } else if ((arg == "-c" || arg == "--ctx-size") && i + 1 < argc) {
            config.n_ctx = std::atoi(argv[++i]);
        } else if ((arg == "-ngl" || arg == "--n-gpu-layers") && i + 1 < argc) {
            config.n_gpu_layers = std::atoi(argv[++i]);
        } else if (arg == "--host" && i + 1 < argc) {
            config.host = argv[++i];
        } else if (arg == "--port" && i + 1 < argc) {
            config.port = std::atoi(argv[++i]);
        } else if (arg == "--max-seq" && i + 1 < argc) {
            config.max_sequences = std::atoi(argv[++i]);
        } else if ((arg == "-b" || arg == "--batch-size") && i + 1 < argc) {
            config.n_batch = std::atoi(argv[++i]);
        } else if (arg == "--admin-token" && i + 1 < argc) {
            config.admin_token = argv[++i];
        } else if (arg == "--log-level" && i + 1 < argc) {
            config.log_level = argv[++i];
        } else {
            print_usage(argv[0]);
            return 1;
        }
    }

    if (config.model_path.empty()) {
        print_usage(argv[0]);
        return 1;
    }

    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);

    // Apply log level from config
    if      (config.log_level == "debug") radixforge::set_log_level(radixforge::LogLevel::DEBUG);
    else if (config.log_level == "warn")  radixforge::set_log_level(radixforge::LogLevel::WARN);
    else if (config.log_level == "error") radixforge::set_log_level(radixforge::LogLevel::ERROR);
    else                                   radixforge::set_log_level(radixforge::LogLevel::INFO);

    RF_INFO("main", "Loading model: %s", config.model_path.c_str());
    RF_INFO("main", "Context: %d tokens, GPU layers: %d, Max sequences: %d",
            config.n_ctx, config.n_gpu_layers, config.max_sequences);

    // Phase 1: Initialize llama.cpp bridge
    radixforge::LlamaBridge bridge(config);
    RF_INFO("main", "Model loaded. KV cache: %d slots", bridge.n_ctx());

    // Phase 2: Initialize Radix Tree
    radixforge::RadixTree tree;

    // Phase 3: Initialize KV Mapper
    radixforge::KVMapper mapper(bridge, tree, config);

    // Phase 4: Start inference worker and HTTP server
    radixforge::InferenceWorker worker(bridge, tree, mapper);
    worker.start();

    radixforge::Server server(config, worker, mapper, tree);
    g_server = &server;

    server.start();  // blocks until shutdown

    worker.stop();
    RF_INFO("main", "Shutdown complete.");
    return 0;
}
