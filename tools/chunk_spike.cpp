#include <llama.h>

#include "radixforge/json_minimal.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

constexpr llama_seq_id kPrefixSeq = 0;
constexpr llama_seq_id kTargetSeq = 1;
constexpr llama_seq_id kTemporarySeq = 2;
constexpr llama_seq_id kScratchSeq = 3;
constexpr double kSanityKlLimit = 1e-3;

const std::string kPrefix =
    "<|im_start|>system\nAnswer the question using the documents. Reply with a "
    "short answer only.<|im_end|>\n<|im_start|>user\nDocuments:\n";

struct Options {
    std::string model_path;
    std::string data_path = "data/hotpot_dev_100.jsonl";
    std::string output_path;
    std::size_t count = 100;
    int gpu_layers = 99;
    std::vector<std::string> variants;
};

struct Item {
    std::string id;
    std::string question;
    std::string answer;
    std::vector<std::string> chunks;
    std::vector<bool> gold;
    std::vector<double> bm25;
};

struct Blob {
    std::vector<uint8_t> data;
    int token_count = 0;
};

struct ItemCache {
    std::vector<Blob> naive;
    std::vector<Blob> prefixed;
    double precompute_ms = 0.0;
    std::size_t naive_bytes = 0;
    std::size_t prefixed_bytes = 0;
};

struct Result {
    std::string variant;
    double kl = 0.0;
    bool top1 = false;
    double em = 0.0;
    double f1 = 0.0;
    bool agree_full = false;
    double ttft_ms = 0.0;
    int tokens_computed = 0;
    int decode_calls = 0;
    double attach_ms = 0.0;
    double recompute_ms = 0.0;
    double suffix_ms = 0.0;
    std::string answer;
    std::vector<float> logits;
};

struct Aggregate {
    std::vector<double> em;
    std::vector<double> f1;
    std::vector<double> agreement;
    std::vector<double> top1;
    std::vector<double> kl;
    std::vector<double> ttft;
    std::vector<int> tokens;
    std::vector<int> decode_calls;
};

[[noreturn]] void fail(const std::string& message) {
    throw std::runtime_error(message);
}

std::vector<std::string> split(std::string_view value, char separator) {
    std::vector<std::string> values;
    std::stringstream stream{std::string(value)};
    std::string part;
    while (std::getline(stream, part, separator)) {
        if (!part.empty()) {
            values.push_back(part);
        }
    }
    return values;
}

Options parse_options(int argc, char** argv) {
    Options options;
    const std::string default_variants =
        "full,naive,prefixed,prefixed+r8,prefixed+r16,prefixed+r32,"
        "prefixed+r64,naive+r16,prefixed+rALL";
    options.variants = split(default_variants, ',');
    for (int i = 1; i < argc; ++i) {
        const std::string argument = argv[i];
        const auto take_value = [&]() -> std::string {
            if (++i >= argc) {
                fail("missing value for " + argument);
            }
            return argv[i];
        };
        if (argument == "-m" || argument == "--model") {
            options.model_path = take_value();
        } else if (argument == "--data") {
            options.data_path = take_value();
        } else if (argument == "--out") {
            options.output_path = take_value();
        } else if (argument == "--n") {
            options.count = static_cast<std::size_t>(std::stoul(take_value()));
        } else if (argument == "--variants") {
            options.variants = split(take_value(), ',');
        } else if (argument == "-ngl" || argument == "--gpu-layers") {
            options.gpu_layers = std::stoi(take_value());
        } else if (argument == "-h" || argument == "--help") {
            std::cout << "Usage: chunk_spike -m MODEL.gguf [--data FILE] [--n N] "
                         "[--variants LIST] [--out FILE] [-ngl 99]\n";
            std::exit(0);
        } else {
            fail("unknown argument: " + argument);
        }
    }
    if (options.model_path.empty()) {
        fail("-m MODEL.gguf is required");
    }
    if (options.output_path.empty()) {
        fail("--out FILE is required");
    }
    if (options.variants.empty()) {
        fail("at least one variant is required");
    }
    return options;
}

std::vector<llama_token> tokenize(
    const llama_vocab* vocab, const std::string& text, bool add_special) {
    std::vector<llama_token> tokens(text.size() + 32U);
    int32_t count = llama_tokenize(vocab, text.c_str(),
                                   static_cast<int32_t>(text.size()), tokens.data(),
                                   static_cast<int32_t>(tokens.size()), add_special, true);
    if (count < 0) {
        tokens.resize(static_cast<std::size_t>(-count));
        count = llama_tokenize(vocab, text.c_str(),
                               static_cast<int32_t>(text.size()), tokens.data(),
                               static_cast<int32_t>(tokens.size()), add_special, true);
    }
    if (count < 0) {
        fail("tokenization failed");
    }
    tokens.resize(static_cast<std::size_t>(count));
    return tokens;
}

bool decode(llama_context* context, const std::vector<llama_token>& tokens,
            llama_seq_id sequence, int position) {
    if (tokens.empty()) {
        return true;
    }
    llama_batch batch = llama_batch_init(static_cast<int32_t>(tokens.size()), 0, 1);
    batch.n_tokens = static_cast<int32_t>(tokens.size());
    for (int32_t i = 0; i < batch.n_tokens; ++i) {
        batch.token[i] = tokens[static_cast<std::size_t>(i)];
        batch.pos[i] = position + i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = sequence;
        batch.logits[i] = i == batch.n_tokens - 1 ? 1 : 0;
    }
    const int status = llama_decode(context, batch);
    llama_batch_free(batch);
    if (status != 0) {
        std::ostringstream message;
        message << "llama_decode failed with status " << status << " at position "
                << position;
        fail(message.str());
    }
    return true;
}

std::vector<float> current_logits(llama_context* context, int vocabulary_size) {
    llama_synchronize(context);
    const float* logits = llama_get_logits_ith(context, -1);
    if (logits == nullptr) {
        fail("llama.cpp did not return logits");
    }
    return {logits, logits + vocabulary_size};
}

void remove_sequence(llama_memory_t memory, llama_seq_id sequence) {
    if (!llama_memory_seq_rm(memory, sequence, -1, -1)) {
        fail("could not remove sequence from KV memory");
    }
}

Blob snapshot(llama_context* context, llama_seq_id sequence, int token_count) {
    const size_t size = llama_state_seq_get_size_ext(
        context, sequence, LLAMA_STATE_SEQ_FLAGS_NONE);
    if (size == 0U) {
        fail("empty state blob");
    }
    Blob blob;
    blob.data.resize(size);
    blob.token_count = token_count;
    const size_t written = llama_state_seq_get_data_ext(
        context, blob.data.data(), blob.data.size(), sequence,
        LLAMA_STATE_SEQ_FLAGS_NONE);
    if (written != blob.data.size()) {
        fail("state snapshot size mismatch");
    }
    return blob;
}

void restore(llama_context* context, const Blob& blob, llama_seq_id sequence) {
    remove_sequence(llama_get_memory(context), sequence);
    const size_t read = llama_state_seq_set_data_ext(
        context, blob.data.data(), blob.data.size(), sequence,
        LLAMA_STATE_SEQ_FLAGS_NONE);
    if (read != blob.data.size()) {
        fail("state restore size mismatch");
    }
}

std::string json_escape(const std::string& value) {
    std::ostringstream escaped;
    for (const unsigned char character : value) {
        switch (character) {
            case '"': escaped << "\\\""; break;
            case '\\': escaped << "\\\\"; break;
            case '\b': escaped << "\\b"; break;
            case '\f': escaped << "\\f"; break;
            case '\n': escaped << "\\n"; break;
            case '\r': escaped << "\\r"; break;
            case '\t': escaped << "\\t"; break;
            default:
                if (character < 0x20U) {
                    escaped << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                            << static_cast<int>(character) << std::dec << std::setfill(' ');
                } else {
                    escaped << character;
                }
        }
    }
    return escaped.str();
}

Item read_item(const std::string& line) {
    const radixforge::json::JsonValue json = radixforge::json::parse(line);
    Item item;
    item.id = json["id"].as_string();
    item.question = json["question"].as_string();
    item.answer = json["answer"].as_string();
    for (const auto& chunk : json["chunks"].as_array()) {
        item.chunks.push_back(chunk.as_string());
    }
    for (const auto& is_gold : json["gold"].as_array()) {
        item.gold.push_back(is_gold.as_bool());
    }
    for (const auto& score : json["bm25"].as_array()) {
        item.bm25.push_back(score.as_number());
    }
    if (item.id.empty() || item.chunks.size() != 10U ||
        item.gold.size() != item.chunks.size() ||
        item.bm25.size() != item.chunks.size()) {
        fail("invalid input item (expected id and 10 chunks with gold and bm25 arrays)");
    }
    return item;
}

std::vector<Item> read_items(const std::string& path, std::size_t count) {
    std::ifstream input(path);
    if (!input) {
        fail("cannot open data file: " + path);
    }
    std::vector<Item> items;
    std::string line;
    while (items.size() < count && std::getline(input, line)) {
        if (!line.empty()) {
            items.push_back(read_item(line));
        }
    }
    if (items.size() != count) {
        fail("data file contains fewer items than --n");
    }
    return items;
}

std::string normalize_answer(const std::string& answer) {
    std::string cleaned;
    for (const unsigned char character : answer) {
        if (std::ispunct(character) == 0) {
            cleaned.push_back(static_cast<char>(std::tolower(character)));
        }
    }
    std::istringstream words(cleaned);
    std::string word;
    std::vector<std::string> retained;
    while (words >> word) {
        if (word != "a" && word != "an" && word != "the") {
            retained.push_back(word);
        }
    }
    std::ostringstream normalized;
    for (std::size_t i = 0; i < retained.size(); ++i) {
        if (i != 0U) {
            normalized << ' ';
        }
        normalized << retained[i];
    }
    return normalized.str();
}

double token_f1(const std::string& prediction, const std::string& gold) {
    const std::vector<std::string> predicted = split(normalize_answer(prediction), ' ');
    const std::vector<std::string> expected = split(normalize_answer(gold), ' ');
    if (predicted.empty() || expected.empty()) {
        return predicted.empty() && expected.empty() ? 1.0 : 0.0;
    }
    std::map<std::string, int> counts;
    for (const std::string& token : expected) {
        ++counts[token];
    }
    int shared = 0;
    for (const std::string& token : predicted) {
        auto iterator = counts.find(token);
        if (iterator != counts.end() && iterator->second > 0) {
            ++shared;
            --iterator->second;
        }
    }
    const double precision = static_cast<double>(shared) / predicted.size();
    const double recall = static_cast<double>(shared) / expected.size();
    return shared == 0 ? 0.0 : 2.0 * precision * recall / (precision + recall);
}

int argmax(const std::vector<float>& logits) {
    return static_cast<int>(std::distance(
        logits.begin(), std::max_element(logits.begin(), logits.end())));
}

double kl_divergence(const std::vector<float>& reference,
                     const std::vector<float>& candidate) {
    const float ref_max = *std::max_element(reference.begin(), reference.end());
    const float candidate_max = *std::max_element(candidate.begin(), candidate.end());
    double ref_sum = 0.0;
    double candidate_sum = 0.0;
    for (std::size_t i = 0; i < reference.size(); ++i) {
        ref_sum += std::exp(static_cast<double>(reference[i] - ref_max));
        candidate_sum += std::exp(static_cast<double>(candidate[i] - candidate_max));
    }
    const double ref_log_z = static_cast<double>(ref_max) + std::log(ref_sum);
    const double candidate_log_z = static_cast<double>(candidate_max) + std::log(candidate_sum);
    double result = 0.0;
    for (std::size_t i = 0; i < reference.size(); ++i) {
        const double log_p = static_cast<double>(reference[i]) - ref_log_z;
        const double log_q = static_cast<double>(candidate[i]) - candidate_log_z;
        result += std::exp(log_p) * (log_p - log_q);
    }
    return result;
}

std::string detokenize(const llama_vocab* vocab, const std::vector<llama_token>& tokens) {
    if (tokens.empty()) {
        return "";
    }
    std::vector<char> output(tokens.size() * 16U + 256U);
    int32_t size = llama_detokenize(vocab, tokens.data(),
                                    static_cast<int32_t>(tokens.size()), output.data(),
                                    static_cast<int32_t>(output.size()), true, true);
    if (size < 0) {
        output.resize(static_cast<std::size_t>(-size));
        size = llama_detokenize(vocab, tokens.data(),
                                static_cast<int32_t>(tokens.size()), output.data(),
                                static_cast<int32_t>(output.size()), true, true);
    }
    if (size < 0) {
        fail("detokenization failed");
    }
    return {output.data(), static_cast<std::size_t>(size)};
}

std::string greedy_answer(llama_context* context, const llama_vocab* vocab,
                          llama_seq_id sequence, int next_position,
                          const std::vector<float>& initial_logits) {
    std::vector<float> logits = initial_logits;
    std::vector<llama_token> generated;
    for (int step = 0; step < 32; ++step) {
        const llama_token token = static_cast<llama_token>(argmax(logits));
        if (llama_vocab_is_eog(vocab, token)) {
            break;
        }
        generated.push_back(token);
        decode(context, {token}, sequence, next_position++);
        logits = current_logits(context, llama_vocab_n_tokens(vocab));
    }
    return detokenize(vocab, generated);
}

struct VariantSpec {
    bool full = false;
    bool prefixed = false;
    int head = 0;          // tokens recomputed at the head of every non-first chunk
    bool all = false;      // recompute every chunk token (sanity check)
    int top_k = 0;         // recompute the top-k chunks by BM25 fully
    bool gold = false;     // recompute the supporting (gold) chunks fully
};

VariantSpec parse_variant(const std::string& variant) {
    VariantSpec spec;
    if (variant == "full") {
        spec.full = true;
        return spec;
    }
    const std::vector<std::string> parts = split(variant, '+');
    if (parts.empty() || (parts[0] != "naive" && parts[0] != "prefixed")) {
        fail("unknown variant: " + variant);
    }
    spec.prefixed = parts[0] == "prefixed";
    for (std::size_t i = 1; i < parts.size(); ++i) {
        const std::string& part = parts[i];
        if (part == "rALL") {
            spec.all = true;
        } else if (part == "gold") {
            spec.gold = true;
        } else if (part.size() > 1U && part[0] == 'r') {
            spec.head = std::stoi(part.substr(1));
        } else if (part.size() > 3U && part.rfind("top", 0) == 0U) {
            spec.top_k = std::stoi(part.substr(3));
        } else {
            fail("unknown variant part '" + part + "' in " + variant);
        }
        if (spec.head < 0 || spec.top_k < 0) {
            fail("negative amount in variant " + variant);
        }
    }
    return spec;
}

std::vector<bool> selected_chunks(const Item& item, const VariantSpec& spec) {
    std::vector<bool> selected(item.chunks.size(), false);
    if (spec.gold) {
        selected = item.gold;
    }
    if (spec.top_k > 0) {
        std::vector<std::size_t> order(item.chunks.size());
        std::iota(order.begin(), order.end(), 0U);
        std::stable_sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
            return item.bm25[a] > item.bm25[b];
        });
        for (std::size_t k = 0; k < order.size() && k < static_cast<std::size_t>(spec.top_k);
             ++k) {
            selected[order[k]] = true;
        }
    }
    return selected;
}

ItemCache precompute_chunks(llama_context* context, llama_memory_t memory,
                            const std::vector<llama_token>& prefix,
                            const std::vector<std::vector<llama_token>>& chunks) {
    ItemCache cache;
    const auto start = std::chrono::steady_clock::now();
    for (const auto& chunk : chunks) {
        decode(context, chunk, kScratchSeq, 0);
        cache.naive.push_back(snapshot(context, kScratchSeq,
                                       static_cast<int>(chunk.size())));
        remove_sequence(memory, kScratchSeq);

        llama_memory_seq_cp(memory, kPrefixSeq, kScratchSeq, -1, -1);
        decode(context, chunk, kScratchSeq, static_cast<int>(prefix.size()));
        cache.prefixed.push_back(snapshot(context, kScratchSeq,
                                          static_cast<int>(chunk.size())));
        remove_sequence(memory, kScratchSeq);
    }
    llama_synchronize(context);
    const auto end = std::chrono::steady_clock::now();
    cache.precompute_ms = std::chrono::duration<double, std::milli>(end - start).count();
    for (const Blob& blob : cache.naive) {
        cache.naive_bytes += blob.data.size();
    }
    for (const Blob& blob : cache.prefixed) {
        cache.prefixed_bytes += blob.data.size();
    }
    return cache;
}

void attach_tail(llama_context* context, llama_memory_t memory, const Blob& blob,
                 bool blob_has_prefix, int prefix_tokens, int recomputed,
                 int final_position) {
    restore(context, blob, kTemporarySeq);
    const int blob_start = blob_has_prefix ? prefix_tokens : 0;
    if (recomputed > 0) {
        if (!llama_memory_seq_rm(memory, kTemporarySeq, blob_start,
                                 blob_start + recomputed)) {
            fail("could not remove recomputed cached head");
        }
    }
    if (blob_has_prefix &&
        !llama_memory_seq_rm(memory, kTemporarySeq, 0, prefix_tokens)) {
        fail("could not remove prefix from imported chunk blob");
    }
    const int tail_start = blob_start + recomputed;
    if (tail_start < blob_start + blob.token_count) {
        llama_memory_seq_add(memory, kTemporarySeq, tail_start, -1,
                             final_position + recomputed - tail_start);
        llama_memory_seq_cp(memory, kTemporarySeq, kTargetSeq, -1, -1);
    }
    remove_sequence(memory, kTemporarySeq);
}

std::vector<float> probe_blob(llama_context* context, llama_memory_t memory,
                              const Blob& blob, int next_position,
                              llama_token probe_token, int vocabulary_size) {
    restore(context, blob, kTemporarySeq);
    decode(context, {probe_token}, kTemporarySeq, next_position);
    const std::vector<float> logits = current_logits(context, vocabulary_size);
    remove_sequence(memory, kTemporarySeq);
    return logits;
}

double max_abs_difference(const std::vector<float>& lhs, const std::vector<float>& rhs) {
    double maximum = 0.0;
    for (std::size_t i = 0; i < lhs.size(); ++i) {
        maximum = std::max(maximum, std::abs(static_cast<double>(lhs[i] - rhs[i])));
    }
    return maximum;
}

Result run_variant(llama_context* context, llama_memory_t memory, const llama_vocab* vocab,
                   const Item& item, const std::vector<llama_token>& prefix,
                   const std::vector<std::vector<llama_token>>& chunks,
                   const std::vector<llama_token>& suffix, const ItemCache& cache,
                   const std::string& variant, const std::vector<float>& full_logits,
                   const std::string& full_answer) {
    const VariantSpec spec = parse_variant(variant);
    const std::vector<bool> selected = selected_chunks(item, spec);
    remove_sequence(memory, kTargetSeq);
    llama_memory_seq_cp(memory, kPrefixSeq, kTargetSeq, -1, -1);
    int position = static_cast<int>(prefix.size());
    int computed = static_cast<int>(suffix.size());
    Result result;
    using Clock = std::chrono::steady_clock;
    const auto ms = [](Clock::time_point from, Clock::time_point to) {
        return std::chrono::duration<double, std::milli>(to - from).count();
    };
    // Tokens to recompute are buffered and decoded in as few llama_decode calls as
    // possible: a flush is only needed before attaching cached cells (positions must
    // stay consecutive per sequence) and the suffix is merged into the last flush.
    std::vector<llama_token> pending;
    int pending_position = position;
    const auto flush = [&]() {
        if (pending.empty()) {
            return;
        }
        const auto t0 = Clock::now();
        decode(context, pending, kTargetSeq, pending_position);
        llama_synchronize(context);
        result.recompute_ms += ms(t0, Clock::now());
        ++result.decode_calls;
        pending.clear();
    };
    const auto queue = [&](std::vector<llama_token>::const_iterator first,
                           std::vector<llama_token>::const_iterator last, int at) {
        if (pending.empty()) {
            pending_position = at;
        }
        pending.insert(pending.end(), first, last);
    };
    const auto start = Clock::now();
    const std::vector<Blob>& blobs = spec.prefixed ? cache.prefixed : cache.naive;
    for (std::size_t index = 0; index < chunks.size(); ++index) {
        const auto& chunk = chunks[index];
        const int size = static_cast<int>(chunk.size());
        int recomputed = 0;
        if (spec.full || spec.all || selected[index]) {
            recomputed = size;
        } else if (index != 0U) {
            recomputed = std::min(spec.head, size);
        }
        if (recomputed > 0) {
            queue(chunk.begin(), chunk.begin() + recomputed, position);
            computed += recomputed;
        }
        if (recomputed < size) {
            flush();
            const auto t0 = Clock::now();
            attach_tail(context, memory, blobs[index], spec.prefixed,
                        static_cast<int>(prefix.size()), recomputed, position);
            result.attach_ms += ms(t0, Clock::now());
        }
        position += size;
    }
    const auto suffix_start = Clock::now();
    queue(suffix.begin(), suffix.end(), position);
    decode(context, pending, kTargetSeq, pending_position);
    llama_synchronize(context);
    ++result.decode_calls;
    pending.clear();
    const auto end = Clock::now();
    // The final call carries buffered recompute tokens plus the suffix (and applies any
    // pending K-shift); it is reported as suffix time.
    result.suffix_ms = ms(suffix_start, end);
    const std::vector<float> logits = current_logits(context, llama_vocab_n_tokens(vocab));
    result.variant = variant;
    result.kl = full_logits.empty() ? 0.0 : kl_divergence(full_logits, logits);
    result.top1 = full_logits.empty() || argmax(full_logits) == argmax(logits);
    result.ttft_ms = ms(start, end);
    result.tokens_computed = computed;
    result.answer = greedy_answer(context, vocab, kTargetSeq,
                                  position + static_cast<int>(suffix.size()), logits);
    result.em = normalize_answer(result.answer) == normalize_answer(item.answer) ? 1.0 : 0.0;
    result.f1 = token_f1(result.answer, item.answer);
    result.agree_full = full_answer.empty() ||
                         normalize_answer(result.answer) == normalize_answer(full_answer);
    result.logits = logits;
    remove_sequence(memory, kTargetSeq);
    return result;
}

void write_result(std::ofstream& output, const Item& item, const Result& result,
                  const ItemCache& cache, int total_chunk_tokens, double independence) {
    const bool prefixed = result.variant.rfind("prefixed", 0) == 0U;
    const std::size_t blob_bytes = prefixed ? cache.prefixed_bytes : cache.naive_bytes;
    output << std::setprecision(12) << "{\"id\":\"" << json_escape(item.id)
           << "\",\"variant\":\"" << json_escape(result.variant)
           << "\",\"kl\":" << result.kl << ",\"top1\":"
           << (result.top1 ? "true" : "false") << ",\"em\":" << result.em
           << ",\"f1\":" << result.f1 << ",\"agree_with_full\":"
           << (result.agree_full ? "true" : "false") << ",\"ttft_ms\":"
           << result.ttft_ms << ",\"tokens_computed\":" << result.tokens_computed
           << ",\"decode_calls\":" << result.decode_calls
           << ",\"attach_ms\":" << result.attach_ms
           << ",\"recompute_ms\":" << result.recompute_ms
           << ",\"suffix_ms\":" << result.suffix_ms
           << ",\"chunk_precompute_ms\":" << cache.precompute_ms
           << ",\"blob_bytes_per_chunk_token\":"
           << static_cast<double>(blob_bytes) / std::max(1, total_chunk_tokens)
           << ",\"independence_max_abs\":" << independence
           << ",\"answer\":\"" << json_escape(result.answer) << "\"}\n";
}

double mean(const std::vector<double>& values) {
    return std::accumulate(values.begin(), values.end(), 0.0) / values.size();
}

double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    const std::size_t middle = values.size() / 2U;
    return values.size() % 2U == 0U ? (values[middle - 1U] + values[middle]) / 2.0
                                    : values[middle];
}

void print_aggregates(const std::vector<std::string>& variants,
                      const std::map<std::string, Aggregate>& aggregates) {
    std::cout << std::fixed << std::setprecision(4)
              << "variant | EM | F1 | agree-with-full | top1 | mean KL | median TTFT ms "
                 "| tokens computed | decode calls\n";
    for (const std::string& variant : variants) {
        const Aggregate& aggregate = aggregates.at(variant);
        const double average_tokens = std::accumulate(
            aggregate.tokens.begin(), aggregate.tokens.end(), 0.0) /
                                      aggregate.tokens.size();
        std::cout << variant << " | " << mean(aggregate.em) << " | " << mean(aggregate.f1)
                  << " | " << mean(aggregate.agreement) << " | " << mean(aggregate.top1)
                  << " | " << mean(aggregate.kl) << " | " << median(aggregate.ttft)
                  << " | " << average_tokens << " | "
                  << std::accumulate(aggregate.decode_calls.begin(),
                                     aggregate.decode_calls.end(), 0.0) /
                         std::max<std::size_t>(1U, aggregate.decode_calls.size())
                  << '\n';
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        const std::vector<Item> items = read_items(options.data_path, options.count);
        llama_backend_init();
        llama_model_params model_params = llama_model_default_params();
        model_params.n_gpu_layers = options.gpu_layers;
        llama_model* model = llama_model_load_from_file(options.model_path.c_str(), model_params);
        if (model == nullptr) {
            fail("failed to load model: " + options.model_path);
        }
        const llama_vocab* vocab = llama_model_get_vocab(model);
        llama_context_params context_params = llama_context_default_params();
        context_params.n_ctx = 8192;
        context_params.n_batch = 8192;
        context_params.n_ubatch = 8192;
        context_params.n_seq_max = 4;
        context_params.kv_unified = true;
        context_params.type_k = GGML_TYPE_F16;
        context_params.type_v = GGML_TYPE_F16;
        llama_context* context = llama_init_from_model(model, context_params);
        if (context == nullptr) {
            llama_model_free(model);
            fail("failed to create context");
        }
        llama_memory_t memory = llama_get_memory(context);
        if (!llama_memory_can_shift(memory)) {
            llama_free(context);
            llama_model_free(model);
            fail("the model's llama.cpp memory implementation cannot shift KV positions");
        }
        std::ofstream output(options.output_path);
        if (!output) {
            llama_free(context);
            llama_model_free(model);
            fail("cannot write output file: " + options.output_path);
        }
        std::map<std::string, Aggregate> aggregates;
        for (const std::string& variant : options.variants) {
            aggregates.emplace(variant, Aggregate{});
        }
        const std::vector<llama_token> prefix = tokenize(vocab, kPrefix, true);
        decode(context, prefix, kPrefixSeq, 0);
        for (std::size_t item_index = 0; item_index < items.size(); ++item_index) {
            const Item& item = items[item_index];
            std::vector<std::vector<llama_token>> chunks;
            int total_chunk_tokens = 0;
            for (std::size_t chunk_index = 0; chunk_index < item.chunks.size(); ++chunk_index) {
                const std::string text = "[" + std::to_string(chunk_index + 1U) + "] " +
                                         item.chunks[chunk_index] + "\n";
                chunks.push_back(tokenize(vocab, text, false));
                total_chunk_tokens += static_cast<int>(chunks.back().size());
            }
            const std::string suffix_text = "Question: " + item.question +
                                            "<|im_end|>\n<|im_start|>assistant\n";
            const std::vector<llama_token> suffix = tokenize(vocab, suffix_text, false);
            const ItemCache cache = precompute_chunks(context, memory, prefix, chunks);

            // Explicitly verify that shifting an imported temporary sequence did not mutate
            // the canonical host blob: restore, decode the same probe, shift/attach a second
            // import, then restore and probe again.
            const llama_token probe = suffix.front();
            const int prefixed_end = static_cast<int>(prefix.size() + chunks.front().size());
            const std::vector<float> before = probe_blob(
                context, memory, cache.prefixed.front(), prefixed_end, probe,
                llama_vocab_n_tokens(vocab));
            llama_memory_seq_cp(memory, kPrefixSeq, kTargetSeq, -1, -1);
            attach_tail(context, memory, cache.prefixed.front(), true,
                        static_cast<int>(prefix.size()), 0,
                        static_cast<int>(prefix.size()) + 7);
            remove_sequence(memory, kTargetSeq);
            const std::vector<float> after = probe_blob(
                context, memory, cache.prefixed.front(), prefixed_end, probe,
                llama_vocab_n_tokens(vocab));
            const double independence = max_abs_difference(before, after);

            Result full = run_variant(context, memory, vocab, item, prefix, chunks, suffix,
                                      cache, "full", {}, "");
            // The shared prefix is excluded from TTFT, but full still reports all
            // request tokens as required by the comparison metric.
            full.tokens_computed += static_cast<int>(prefix.size());
            std::map<std::string, Result> results;
            results.emplace("full", std::move(full));
            for (const std::string& variant : options.variants) {
                if (variant == "full") {
                    continue;
                }
                Result result = run_variant(context, memory, vocab, item, prefix, chunks,
                                            suffix, cache, variant,
                                            results.at("full").logits,
                                            results.at("full").answer);
                if (variant == "prefixed+rALL" && result.kl > kSanityKlLimit) {
                    std::ostringstream message;
                    message << "rALL sanity failed for " << item.id << ": KL=" << result.kl;
                    fail(message.str());
                }
                results.emplace(variant, std::move(result));
            }
            for (const std::string& variant : options.variants) {
                const Result& result = results.at(variant);
                Aggregate& aggregate = aggregates.at(variant);
                aggregate.em.push_back(result.em);
                aggregate.f1.push_back(result.f1);
                aggregate.agreement.push_back(result.agree_full ? 1.0 : 0.0);
                aggregate.top1.push_back(result.top1 ? 1.0 : 0.0);
                aggregate.kl.push_back(result.kl);
                aggregate.ttft.push_back(result.ttft_ms);
                aggregate.tokens.push_back(result.tokens_computed);
                aggregate.decode_calls.push_back(result.decode_calls);
                write_result(output, item, result, cache, total_chunk_tokens, independence);
            }
            std::cout << "completed " << (item_index + 1U) << "/" << items.size()
                      << " (independence max abs " << independence << ")\n";
        }
        print_aggregates(options.variants, aggregates);
        llama_free(context);
        llama_model_free(model);
        llama_backend_free();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "chunk_spike: " << error.what() << '\n';
        llama_backend_free();
        return 1;
    }
}
