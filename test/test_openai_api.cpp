// Unit tests for pure OpenAI API parsing, formatting, and the minimal JSON parser.

#include "radixforge/json_minimal.h"
#include "radixforge/openai_api.h"

#include <cmath>
#include <cstdio>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>

namespace {

int g_pass = 0;
int g_fail = 0;

#define CHECK(condition, message)                                                                  \
    do {                                                                                           \
        if (condition) {                                                                           \
            ++g_pass;                                                                              \
            std::printf("  PASS  %s\n", message);                                                  \
        } else {                                                                                   \
            ++g_fail;                                                                              \
            std::printf("  FAIL  %s (line %d)\n", message, __LINE__);                              \
        }                                                                                          \
    } while (0)

void expect_throws(const std::function<void()>& operation, const char* message) {
    try {
        operation();
        CHECK(false, message);
    } catch (const std::exception&) {
        CHECK(true, message);
    }
}

radixforge::Config default_config() {
    radixforge::Config config;
    config.max_tokens = 123;
    config.temperature = 0.25F;
    config.top_p = 0.75F;
    return config;
}

void test_chat_request_parsing() {
    std::printf("\n[test_chat_request_parsing]\n");
    radixforge::InferenceRequest request;
    radixforge::openai_api::parse_request_body(
        R"({"messages":[{"role":"system","content":"rules"},{"role":"user","content":"hi"}],"max_tokens":7,"temperature":0,"top_p":1,"stream":true})",
        request, default_config());

    CHECK(request.messages.size() == 2, "parses chat message array");
    CHECK(request.messages[0].role == "system" && request.messages[0].content == "rules",
          "preserves first chat role and content");
    CHECK(request.messages[1].role == "user" && request.messages[1].content == "hi",
          "preserves second chat role and content");
    CHECK(request.prompt.empty(), "chat request leaves prompt empty");
    CHECK(request.max_tokens == 7, "parses max_tokens");
    CHECK(request.temperature == 0.0F && request.top_p == 1.0F, "parses generation numeric values");
    CHECK(request.stream, "parses stream=true");
}

void test_completion_defaults_and_stream() {
    std::printf("\n[test_completion_defaults_and_stream]\n");
    radixforge::InferenceRequest request;
    radixforge::openai_api::parse_request_body(R"({"prompt":"finish this","stream":false})",
                                               request, default_config());

    CHECK(request.prompt == "finish this", "parses completion prompt");
    CHECK(request.messages.empty(), "completion request has no chat messages");
    CHECK(request.max_tokens == 123, "applies configured max_tokens default");
    CHECK(request.temperature == 0.25F && request.top_p == 0.75F,
          "applies configured generation defaults");
    CHECK(!request.stream, "stream defaults to false when omitted or false");
}

void test_max_tokens_validation() {
    std::printf("\n[test_max_tokens_validation]\n");
    const radixforge::Config config = default_config();
    for (const std::string& value : {"-1", "1.5", "2147483648", "1e9999", "true", "null"}) {
        expect_throws(
            [&] {
                radixforge::InferenceRequest request;
                radixforge::openai_api::parse_request_body(
                    "{\"prompt\":\"x\",\"max_tokens\":" + value + "}", request, config);
            },
            "rejects invalid max_tokens value");
    }

    radixforge::InferenceRequest zero;
    radixforge::openai_api::parse_request_body(R"({"prompt":"x","max_tokens":0})", zero, config);
    CHECK(zero.max_tokens == 0, "accepts zero max_tokens");
}

void test_response_and_error_shapes() {
    std::printf("\n[test_response_and_error_shapes]\n");
    radixforge::InferenceStats stats;
    stats.prompt_n = 3;
    stats.prompt_tokens = 4;
    stats.predicted_n = 2;
    stats.predicted_ms = 1.5;

    const std::string chat = radixforge::openai_api::completion_json("hello\nworld", stats, true);
    const std::string text = radixforge::openai_api::completion_json("hello", stats, false);
    const std::string error =
        radixforge::openai_api::error_json("bad \"request\"", "invalid_request_error");
    const auto chat_json = radixforge::json::parse(chat);
    const auto text_json = radixforge::json::parse(text);
    const auto error_json = radixforge::json::parse(error);

    CHECK(chat_json["object"].as_string() == "chat.completion", "formats chat completion object");
    CHECK(chat_json["choices"][0]["message"]["role"].as_string() == "assistant",
          "formats chat assistant role");
    CHECK(chat_json["choices"][0]["message"]["content"].as_string() == "hello\nworld",
          "escapes and preserves chat content");
    CHECK(text_json["object"].as_string() == "text_completion", "formats text completion object");
    CHECK(text_json["choices"][0]["text"].as_string() == "hello",
          "formats text completion content");
    CHECK(chat_json["usage"]["total_tokens"].as_number() == 6, "formats usage total token count");
    CHECK(error_json["error"]["type"].as_string() == "invalid_request_error",
          "formats OpenAI error type");
    CHECK(error_json["error"]["message"].as_string() == "bad \"request\"",
          "formats escaped OpenAI error message");
}

void test_utf8_boundary_helper() {
    std::printf("\n[test_utf8_boundary_helper]\n");
    const std::string complete = std::string("A") + "\xE2" + "\x82" + "\xAC";
    const std::string incomplete_two = std::string("A") + "\xC2";
    const std::string incomplete_three_one = std::string("A") + "\xE2";
    const std::string incomplete_three_two = std::string("A") + "\xE2" + "\x82";
    const std::string incomplete_four_one = std::string("A") + "\xF0";
    const std::string incomplete_four_two = std::string("A") + "\xF0" + "\x9F";
    const std::string incomplete_four_three = std::string("A") + "\xF0" + "\x9F" + "\x92";
    const std::string invalid_lead = std::string("A") + "\xC0";
    const std::string invalid_extended_lead = std::string("A") + "\xF8";
    const std::string isolated_continuation = std::string("A") + "\x80";
    CHECK(radixforge::openai_api::complete_utf8_prefix_length(complete) == complete.size(),
          "keeps complete multi-byte UTF-8 suffix");
    CHECK(radixforge::openai_api::complete_utf8_prefix_length(incomplete_two) == 1,
          "buffers incomplete two-byte UTF-8 suffix");
    CHECK(radixforge::openai_api::complete_utf8_prefix_length(incomplete_three_one) == 1,
          "buffers one byte of incomplete three-byte UTF-8 suffix");
    CHECK(radixforge::openai_api::complete_utf8_prefix_length(incomplete_three_two) == 1,
          "buffers two bytes of incomplete three-byte UTF-8 suffix");
    CHECK(radixforge::openai_api::complete_utf8_prefix_length(incomplete_four_one) == 1,
          "buffers one byte of incomplete four-byte UTF-8 suffix");
    CHECK(radixforge::openai_api::complete_utf8_prefix_length(incomplete_four_two) == 1,
          "buffers two bytes of incomplete four-byte UTF-8 suffix");
    CHECK(radixforge::openai_api::complete_utf8_prefix_length(incomplete_four_three) == 1,
          "buffers three bytes of incomplete four-byte UTF-8 suffix");
    CHECK(radixforge::openai_api::complete_utf8_prefix_length(invalid_lead) == invalid_lead.size(),
          "flushes invalid UTF-8 lead byte");
    CHECK(radixforge::openai_api::complete_utf8_prefix_length(invalid_extended_lead) ==
              invalid_extended_lead.size(),
          "flushes invalid extended UTF-8 lead byte");
    CHECK(radixforge::openai_api::complete_utf8_prefix_length(isolated_continuation) ==
              isolated_continuation.size(),
          "flushes isolated UTF-8 continuation byte");
}

void test_json_parser_hardening() {
    std::printf("\n[test_json_parser_hardening]\n");
    const auto escaped = radixforge::json::parse(R"({"s":"\"\\\/\b\f\n\r\t","u":"\u20ac"})");
    CHECK(escaped["s"].as_string() == "\"\\/\b\f\n\r\t", "parses standard string escapes");
    CHECK(escaped["u"].as_string() == "\xE2\x82\xAC", "parses Unicode BMP escape as UTF-8");
    const auto surrogate_pair = radixforge::json::parse(R"({"face":"\ud83d\ude00"})");
    CHECK(surrogate_pair["face"].as_string() == "\xF0\x9F\x98\x80",
          "decodes Unicode surrogate pair as UTF-8");
    const auto nested =
        radixforge::json::parse(R"({"outer":[{"escaped":"line\nquote\""}],"n":-1.25e+2})");
    CHECK(nested["outer"][0]["escaped"].as_string() == "line\nquote\"",
          "accepts nested JSON with escapes");
    CHECK(nested["n"].as_number() == -125.0, "accepts JSON numbers with exponents");

    expect_throws([] { radixforge::json::parse("{} trailing"); }, "rejects trailing data");
    for (const std::string& malformed : {"",
                                         "true false",
                                         "[1 2]",
                                         "{",
                                         "[",
                                         "{\"a\":1",
                                         "[1,",
                                         "{\"a\":1,}",
                                         "{\"a\" 1}",
                                         "{\"a\":1 \"b\":2}",
                                         "\"unterminated",
                                         "\"bad\\q\"",
                                         "\"bad\\u12\"",
                                         "\"bad\\u12xz\"",
                                         "\"\\ud800\"",
                                         "\"\\udc00\"",
                                         "01",
                                         "-",
                                         "1.",
                                         "1e",
                                         "1e+",
                                         "[,]",
                                         "fal",
                                         "nul"}) {
        expect_throws([&] { radixforge::json::parse(malformed); }, "rejects malformed JSON input");
    }

    const std::string raw_control = std::string("\"") + '\x01' + "\"";
    expect_throws([&] { radixforge::json::parse(raw_control); },
                  "rejects raw control characters in JSON strings");

    std::string at_limit;
    std::string over_limit;
    for (int i = 0; i < 64; ++i) {
        at_limit += "[";
        over_limit += "[";
    }
    over_limit += "[";
    at_limit += "0";
    over_limit += "0";
    for (int i = 0; i < 64; ++i) {
        at_limit += "]";
        over_limit += "]";
    }
    over_limit += "]";
    CHECK(radixforge::json::parse(at_limit).type() == radixforge::json::Type::Array,
          "accepts JSON nesting depth 64");
    expect_throws([&] { radixforge::json::parse(over_limit); },
                  "rejects JSON nesting depth over 64");
}

} // namespace

int main() {
    std::printf("=== OpenAI API Unit Tests ===\n");
    test_chat_request_parsing();
    test_completion_defaults_and_stream();
    test_max_tokens_validation();
    test_response_and_error_shapes();
    test_utf8_boundary_helper();
    test_json_parser_hardening();
    std::printf("\nResults: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
