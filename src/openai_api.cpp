#include "radixforge/openai_api.h"

#include "radixforge/json_minimal.h"

#include <cmath>
#include <cstdio>
#include <limits>
#include <stdexcept>

namespace radixforge::openai_api {

std::string timings_json(const InferenceStats& stats) {
    return "{\"prompt_n\":" + std::to_string(stats.prompt_n) +
           ",\"cache_n\":" + std::to_string(stats.cache_n) +
           ",\"prompt_ms\":" + std::to_string(stats.prompt_ms) +
           ",\"predicted_n\":" + std::to_string(stats.predicted_n) +
           ",\"predicted_ms\":" + std::to_string(stats.predicted_ms) + "}";
}

std::string usage_json(const InferenceStats& stats) {
    return "{\"prompt_tokens\":" + std::to_string(stats.prompt_tokens) +
           ",\"completion_tokens\":" + std::to_string(stats.predicted_n) +
           ",\"total_tokens\":" + std::to_string(stats.prompt_tokens + stats.predicted_n) + "}";
}

std::string json_escape(const std::string& value) {
    std::string escaped;
    escaped.reserve(value.size() + 16);
    for (unsigned char c : value) {
        switch (c) {
        case '"':
            escaped += "\\\"";
            break;
        case '\\':
            escaped += "\\\\";
            break;
        case '\n':
            escaped += "\\n";
            break;
        case '\r':
            escaped += "\\r";
            break;
        case '\t':
            escaped += "\\t";
            break;
        default:
            if (c < 0x20) {
                char buffer[8];
                std::snprintf(buffer, sizeof(buffer), "\\u%04x", c);
                escaped += buffer;
            } else {
                escaped += static_cast<char>(c);
            }
        }
    }
    return escaped;
}

std::string error_json(const std::string& message, const std::string& type) {
    return "{\"error\":{\"message\":\"" + json_escape(message) + "\",\"type\":\"" + type + "\"}}";
}

void parse_request_body(const std::string& body_text, InferenceRequest& request,
                        const Config& config) {
    const json::JsonValue body = json::parse(body_text);

    if (body.contains("messages") && body["messages"].type() == json::Type::Array) {
        for (size_t i = 0; i < body["messages"].size(); ++i) {
            const auto& message = body["messages"][i];
            ChatMessage chat_message;
            chat_message.role = message.contains("role") ? message["role"].as_string() : "user";
            chat_message.content =
                message.contains("content") ? message["content"].as_string() : "";
            request.messages.push_back(std::move(chat_message));
        }
    } else if (body.contains("prompt") && body["prompt"].type() == json::Type::String) {
        request.prompt = body["prompt"].as_string();
    }

    if (body.contains("max_tokens")) {
        if (body["max_tokens"].type() != json::Type::Number) {
            throw std::invalid_argument("max_tokens must be a number");
        }
        const double max_tokens = body["max_tokens"].as_number();
        if (!std::isfinite(max_tokens) || max_tokens < 0 ||
            max_tokens > std::numeric_limits<int32_t>::max() ||
            std::floor(max_tokens) != max_tokens) {
            throw std::invalid_argument("max_tokens must be a non-negative int32");
        }
        request.max_tokens = static_cast<int32_t>(max_tokens);
    } else {
        request.max_tokens = config.max_tokens;
    }

    request.temperature =
        body.contains("temperature") && body["temperature"].type() == json::Type::Number
            ? static_cast<float>(body["temperature"].as_number())
            : config.temperature;
    request.top_p = body.contains("top_p") && body["top_p"].type() == json::Type::Number
                        ? static_cast<float>(body["top_p"].as_number())
                        : config.top_p;
    request.stream = body.contains("stream") && body["stream"].type() == json::Type::Bool &&
                     body["stream"].as_bool();
}

std::string completion_json(const std::string& content, const InferenceStats& stats, bool is_chat) {
    if (is_chat) {
        return "{\"object\":\"chat.completion\",\"choices\":[{\"message\":{"
               "\"role\":\"assistant\",\"content\":\"" +
               json_escape(content) +
               "\"},\"finish_reason\":\"stop\"}],\"timings\":" + timings_json(stats) +
               ",\"usage\":" + usage_json(stats) + "}";
    }
    return "{\"object\":\"text_completion\",\"choices\":[{\"text\":\"" + json_escape(content) +
           "\",\"finish_reason\":\"stop\"}],\"timings\":" + timings_json(stats) +
           ",\"usage\":" + usage_json(stats) + "}";
}

size_t complete_utf8_prefix_length(const std::string& text) {
    if (text.empty())
        return 0;

    size_t continuation_count = 0;
    size_t suffix_start = text.size();
    while (suffix_start > 0 && continuation_count < 3 &&
           (static_cast<unsigned char>(text[suffix_start - 1]) & 0xC0) == 0x80) {
        --suffix_start;
        ++continuation_count;
    }

    if (continuation_count == 0) {
        const unsigned char last = static_cast<unsigned char>(text.back());
        const size_t expected = last >= 0xF0 && last <= 0xF4   ? 4
                                : last >= 0xE0 && last <= 0xEF ? 3
                                : last >= 0xC2 && last <= 0xDF ? 2
                                                               : 1;
        return expected > 1 ? text.size() - 1 : text.size();
    }

    if (suffix_start == 0)
        return text.size();

    const size_t lead_position = suffix_start - 1;
    const unsigned char lead = static_cast<unsigned char>(text[lead_position]);
    const size_t expected = lead >= 0xF0 && lead <= 0xF4   ? 4
                            : lead >= 0xE0 && lead <= 0xEF ? 3
                            : lead >= 0xC2 && lead <= 0xDF ? 2
                                                           : 1;
    const size_t available = continuation_count + 1;
    return expected > available ? lead_position : text.size();
}

} // namespace radixforge::openai_api
