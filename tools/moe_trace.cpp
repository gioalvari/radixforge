// moe_trace: record the routed experts of every token for a JSONL chat workload.
//
// For each request the tool renders the model's chat template, prefills the
// prompt from an empty cache, decodes greedily, and captures the
// `ffn_moe_argsort-<layer>` graph tensors through `cb_eval`. The first k entries
// of each argsort row are the experts selected for that token in that layer.
//
// Outputs (for `--out PREFIX`):
//   PREFIX.meta.json  model, n_expert, k, MoE layer ids
//   PREFIX.bin        uint8 expert ids, per request [n_routed][n_moe_layers][k]
//   PREFIX.jsonl      one row per request: ids, offsets, tokens, generated text

#include <ggml-backend.h>
#include <ggml.h>
#include <llama.h>

#include "radixforge/json_minimal.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr const char* kTensorPrefix = "ffn_moe_argsort-";
constexpr std::size_t kTensorPrefixLen = 16;

[[noreturn]] void fail(const std::string& message) {
    throw std::runtime_error(message);
}

struct Options {
    std::string model_path;
    std::string data_path;
    std::string out_prefix;
    std::size_t limit = 0; // 0 = all
    int gpu_layers = 99;
    int ctx_size = 8192;
    int max_tokens_override = -1;
};

struct Message {
    std::string role;
    std::string content;
};

struct Request {
    std::string id;
    std::string family;
    std::string agent;
    int turn = 0;
    int max_tokens = 128;
    std::vector<Message> messages;
};

// Routing captured during one llama_decode call (one ubatch).
struct Capture {
    int n_expert = 0;
    int k = 0;
    int n_tokens = 0;
    std::map<int, std::vector<uint8_t>> by_layer; // layer -> [n_tokens][k]
};

bool on_tensor(ggml_tensor* t, bool ask, void* user_data) {
    if (std::strncmp(t->name, kTensorPrefix, kTensorPrefixLen) != 0) {
        return ask ? false : true;
    }
    if (ask) {
        return true;
    }
    auto* cap = static_cast<Capture*>(user_data);
    if (t->type != GGML_TYPE_I32 || !ggml_is_contiguous(t)) {
        fail(std::string("unexpected tensor layout for ") + t->name);
    }
    const int layer = std::atoi(t->name + kTensorPrefixLen);
    const int n_expert = static_cast<int>(t->ne[0]);
    const int n_tokens = static_cast<int>(t->ne[1]);
    if (n_expert > 256) {
        fail("n_expert > 256 does not fit in uint8 trace");
    }
    std::vector<int32_t> ids(static_cast<std::size_t>(n_expert) * n_tokens);
    ggml_backend_tensor_get(t, ids.data(), 0, ids.size() * sizeof(int32_t));

    cap->n_expert = n_expert;
    cap->n_tokens = n_tokens;
    std::vector<uint8_t>& row = cap->by_layer[layer];
    row.resize(static_cast<std::size_t>(n_tokens) * cap->k);
    for (int tok = 0; tok < n_tokens; ++tok) {
        for (int j = 0; j < cap->k; ++j) {
            row[static_cast<std::size_t>(tok) * cap->k + j] =
                static_cast<uint8_t>(ids[static_cast<std::size_t>(tok) * n_expert + j]);
        }
    }
    return true;
}

std::string json_escape(const std::string& s) {
    std::ostringstream out;
    for (const unsigned char c : s) {
        switch (c) {
        case '"':
            out << "\\\"";
            break;
        case '\\':
            out << "\\\\";
            break;
        case '\n':
            out << "\\n";
            break;
        case '\r':
            out << "\\r";
            break;
        case '\t':
            out << "\\t";
            break;
        default:
            if (c < 0x20) {
                char buf[8];
                std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                out << buf;
            } else {
                out << c;
            }
        }
    }
    return out.str();
}

std::vector<Request> read_requests(const Options& opt) {
    std::ifstream input(opt.data_path);
    if (!input) {
        fail("cannot open data file: " + opt.data_path);
    }
    std::vector<Request> out;
    std::string line;
    while (std::getline(input, line)) {
        if (line.empty()) {
            continue;
        }
        const auto json = radixforge::json::parse(line);
        Request r;
        r.id = json["id"].as_string();
        r.family = json["family"].as_string();
        r.agent = json.contains("agent") ? json["agent"].as_string() : r.family;
        r.turn = json.contains("turn") ? static_cast<int>(json["turn"].as_number()) : 0;
        if (json.contains("max_tokens")) {
            r.max_tokens = static_cast<int>(json["max_tokens"].as_number());
        }
        if (opt.max_tokens_override >= 0) {
            r.max_tokens = opt.max_tokens_override;
        }
        for (const auto& m : json["messages"].as_array()) {
            r.messages.push_back({m["role"].as_string(), m["content"].as_string()});
        }
        if (r.id.empty() || r.messages.empty()) {
            fail("request without id or messages: " + line.substr(0, 80));
        }
        out.push_back(std::move(r));
        if (opt.limit != 0 && out.size() >= opt.limit) {
            break;
        }
    }
    return out;
}

std::string render(const llama_model* model, const Request& r) {
    const char* tmpl = llama_model_chat_template(model, nullptr);
    if (tmpl == nullptr) {
        fail("model has no chat template");
    }
    std::vector<llama_chat_message> chat;
    for (const auto& m : r.messages) {
        chat.push_back({m.role.c_str(), m.content.c_str()});
    }
    std::vector<char> buf(16384);
    int32_t n = llama_chat_apply_template(tmpl, chat.data(), chat.size(), true, buf.data(),
                                          static_cast<int32_t>(buf.size()));
    if (n > static_cast<int32_t>(buf.size())) {
        buf.resize(static_cast<std::size_t>(n));
        n = llama_chat_apply_template(tmpl, chat.data(), chat.size(), true, buf.data(), n);
    }
    if (n < 0) {
        fail("chat template failed for " + r.id);
    }
    return std::string(buf.data(), static_cast<std::size_t>(n));
}

std::vector<llama_token> tokenize(const llama_vocab* vocab, const std::string& text) {
    int32_t n = -llama_tokenize(vocab, text.c_str(), static_cast<int32_t>(text.size()), nullptr, 0,
                                true, true);
    std::vector<llama_token> tokens(static_cast<std::size_t>(n));
    if (llama_tokenize(vocab, text.c_str(), static_cast<int32_t>(text.size()), tokens.data(), n,
                       true, true) < 0) {
        fail("tokenize failed");
    }
    return tokens;
}

std::string piece(const llama_vocab* vocab, llama_token token) {
    char buf[256];
    const int32_t n = llama_token_to_piece(vocab, token, buf, sizeof(buf), 0, false);
    return n > 0 ? std::string(buf, static_cast<std::size_t>(n)) : std::string();
}

Options parse_args(int argc, char** argv) {
    Options opt;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) {
                fail("missing value for " + a);
            }
            return argv[++i];
        };
        if (a == "-m" || a == "--model")
            opt.model_path = next();
        else if (a == "--data")
            opt.data_path = next();
        else if (a == "--out")
            opt.out_prefix = next();
        else if (a == "--limit")
            opt.limit = std::stoul(next());
        else if (a == "-ngl")
            opt.gpu_layers = std::stoi(next());
        else if (a == "-c" || a == "--ctx-size")
            opt.ctx_size = std::stoi(next());
        else if (a == "--max-tokens")
            opt.max_tokens_override = std::stoi(next());
        else
            fail("unknown argument: " + a);
    }
    if (opt.model_path.empty() || opt.data_path.empty() || opt.out_prefix.empty()) {
        fail("usage: moe_trace -m MODEL --data WORKLOAD.jsonl --out PREFIX "
             "[--limit N] [--max-tokens N] [-c CTX] [-ngl N]");
    }
    return opt;
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Options opt = parse_args(argc, argv);
        const std::vector<Request> requests = read_requests(opt);

        llama_backend_init();
        llama_model_params mparams = llama_model_default_params();
        mparams.n_gpu_layers = opt.gpu_layers;
        llama_model* model = llama_model_load_from_file(opt.model_path.c_str(), mparams);
        if (model == nullptr) {
            fail("failed to load model");
        }
        const llama_vocab* vocab = llama_model_get_vocab(model);

        char kbuf[32] = {0};
        std::string arch(64, '\0');
        llama_model_meta_val_str(model, "general.architecture", arch.data(), arch.size());
        arch.resize(std::strlen(arch.c_str()));
        if (llama_model_meta_val_str(model, (arch + ".expert_used_count").c_str(), kbuf,
                                     sizeof(kbuf)) < 0) {
            fail("model has no expert_used_count: not a MoE?");
        }

        Capture cap;
        cap.k = std::atoi(kbuf);

        llama_context_params cparams = llama_context_default_params();
        cparams.n_ctx = static_cast<uint32_t>(opt.ctx_size);
        cparams.n_batch = 512;
        cparams.n_ubatch = 512;
        cparams.cb_eval = on_tensor;
        cparams.cb_eval_user_data = &cap;
        llama_context* ctx = llama_init_from_model(model, cparams);
        if (ctx == nullptr) {
            fail("failed to create context");
        }
        llama_sampler* sampler = llama_sampler_init_greedy();

        std::ofstream bin(opt.out_prefix + ".bin", std::ios::binary);
        std::ofstream rows(opt.out_prefix + ".jsonl");
        if (!bin || !rows) {
            fail("cannot open output files for prefix " + opt.out_prefix);
        }

        std::vector<int> moe_layers;
        uint64_t offset = 0;

        // Decode `tokens` starting at `pos`, append routing to `trace`.
        auto decode = [&](const llama_token* tokens, int n, int pos, std::vector<uint8_t>& trace) {
            llama_batch batch = llama_batch_init(n, 0, 1);
            for (int i = 0; i < n; ++i) {
                batch.token[i] = tokens[i];
                batch.pos[i] = pos + i;
                batch.n_seq_id[i] = 1;
                batch.seq_id[i][0] = 0;
                batch.logits[i] = (i == n - 1);
            }
            batch.n_tokens = n;
            cap.by_layer.clear();
            const int rc = llama_decode(ctx, batch);
            llama_batch_free(batch);
            if (rc != 0) {
                fail("llama_decode failed with code " + std::to_string(rc));
            }
            if (cap.n_tokens != n || cap.by_layer.empty()) {
                fail("routing capture mismatch (ubatch split?)");
            }
            if (moe_layers.empty()) {
                for (const auto& [layer, _] : cap.by_layer)
                    moe_layers.push_back(layer);
            } else if (cap.by_layer.size() != moe_layers.size()) {
                fail("MoE layer count changed between decodes");
            }
            const std::size_t k = static_cast<std::size_t>(cap.k);
            for (int tok = 0; tok < n; ++tok) {
                for (const int layer : moe_layers) {
                    const auto& row = cap.by_layer.at(layer);
                    trace.insert(trace.end(), row.begin() + tok * k, row.begin() + (tok + 1) * k);
                }
            }
        };

        for (std::size_t ri = 0; ri < requests.size(); ++ri) {
            const Request& r = requests[ri];
            const std::vector<llama_token> prompt = tokenize(vocab, render(model, r));
            if (static_cast<int>(prompt.size()) + r.max_tokens > opt.ctx_size) {
                std::cerr << "skip " << r.id << ": prompt too long\n";
                continue;
            }
            llama_memory_clear(llama_get_memory(ctx), true);
            llama_sampler_reset(sampler);

            std::vector<uint8_t> trace;
            const auto t0 = std::chrono::steady_clock::now();
            int pos = 0;
            while (pos < static_cast<int>(prompt.size())) {
                const int n = std::min<int>(512, static_cast<int>(prompt.size()) - pos);
                decode(prompt.data() + pos, n, pos, trace);
                pos += n;
            }
            const auto t1 = std::chrono::steady_clock::now();

            std::vector<llama_token> generated;
            std::string text;
            while (static_cast<int>(generated.size()) < r.max_tokens) {
                const llama_token tok = llama_sampler_sample(sampler, ctx, -1);
                llama_sampler_accept(sampler, tok);
                if (llama_vocab_is_eog(vocab, tok)) {
                    break;
                }
                generated.push_back(tok);
                text += piece(vocab, tok);
                if (static_cast<int>(generated.size()) == r.max_tokens) {
                    break; // last token is emitted but never routed
                }
                decode(&tok, 1, pos, trace);
                ++pos;
            }
            const auto t2 = std::chrono::steady_clock::now();

            const std::size_t per_token = moe_layers.size() * static_cast<std::size_t>(cap.k);
            const std::size_t n_routed = trace.size() / per_token;
            bin.write(reinterpret_cast<const char*>(trace.data()),
                      static_cast<std::streamsize>(trace.size()));

            rows << "{\"id\":\"" << json_escape(r.id) << "\",\"family\":\"" << json_escape(r.family)
                 << "\",\"agent\":\"" << json_escape(r.agent) << "\",\"turn\":" << r.turn
                 << ",\"n_prompt\":" << prompt.size() << ",\"n_gen\":" << generated.size()
                 << ",\"n_routed\":" << n_routed << ",\"offset\":" << offset
                 << ",\"prefill_ms\":" << std::chrono::duration<double, std::milli>(t1 - t0).count()
                 << ",\"decode_ms\":" << std::chrono::duration<double, std::milli>(t2 - t1).count()
                 << ",\"tokens\":[";
            for (std::size_t i = 0; i < prompt.size(); ++i)
                rows << (i ? "," : "") << prompt[i];
            for (const llama_token tok : generated)
                rows << "," << tok;
            rows << "],\"text\":\"" << json_escape(text) << "\"}\n";
            rows.flush();
            offset += trace.size();

            std::cerr << "[" << ri + 1 << "/" << requests.size() << "] " << r.id
                      << " prompt=" << prompt.size() << " gen=" << generated.size() << "\n";
        }

        std::ofstream meta(opt.out_prefix + ".meta.json");
        char desc[256] = {0};
        llama_model_desc(model, desc, sizeof(desc));
        meta << "{\"model\":\"" << json_escape(opt.model_path) << "\",\"desc\":\""
             << json_escape(desc) << "\",\"n_expert\":" << cap.n_expert << ",\"k\":" << cap.k
             << ",\"moe_layers\":[";
        for (std::size_t i = 0; i < moe_layers.size(); ++i)
            meta << (i ? "," : "") << moe_layers[i];
        meta << "],\"dtype\":\"uint8\",\"layout\":\"[n_routed][n_moe_layers][k]\"}\n";

        llama_sampler_free(sampler);
        llama_free(ctx);
        llama_model_free(model);
        llama_backend_free();
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "moe_trace: " << e.what() << "\n";
        return 1;
    }
}
