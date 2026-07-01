#include "radixforge/config.h"
#include "radixforge/kv_mapper.h"
#include "radixforge/llama_bridge.h"
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
        "  -m, --model <path>     Path to GGUF model (required)\n"
        "  -c, --ctx-size <n>     Context size (default: 32768)\n"
        "  -b, --batch-size <n>   Max decode batch size (default: 2048)\n"
        "  -ngl, --n-gpu-layers   GPU layers to offload (default: 99)\n"
        "  --host <addr>          Listen address (default: 127.0.0.1)\n"
        "  --port <n>             Listen port (default: 8400)\n"
        "  --max-seq <n>          Max concurrent sequences (default: 32)\n"
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

    fprintf(stderr, "[radixforge] Loading model: %s\n", config.model_path.c_str());
    fprintf(stderr, "[radixforge] Context: %d tokens, GPU layers: %d, Max sequences: %d\n",
            config.n_ctx, config.n_gpu_layers, config.max_sequences);

    // Phase 1: Initialize llama.cpp bridge
    radixforge::LlamaBridge bridge(config);
    fprintf(stderr, "[radixforge] Model loaded successfully. KV cache: %d slots\n",
            bridge.n_ctx());

    // Phase 2: Initialize Radix Tree
    radixforge::RadixTree tree;

    // Phase 3: Initialize KV Mapper
    radixforge::KVMapper mapper(bridge, tree, config);

    // Phase 4: Start inference worker and HTTP server
    radixforge::InferenceWorker worker(bridge, tree, mapper);
    worker.start();

    radixforge::Server server(config, worker);
    g_server = &server;
    fprintf(stderr, "[radixforge] Server listening on %s:%d\n",
            config.host.c_str(), config.port);

    server.start();  // blocks until shutdown

    worker.stop();
    fprintf(stderr, "[radixforge] Shutdown complete.\n");
    return 0;
}
