#pragma once

#include "radixforge/config.h"
#include "radixforge/server.h"

#include <cstddef>
#include <string>

namespace radixforge::openai_api {

std::string json_escape(const std::string& value);
std::string error_json(const std::string& message,
                       const std::string& type = "server_error");
std::string timings_json(const InferenceStats& stats);
std::string usage_json(const InferenceStats& stats);
void parse_request_body(const std::string& body, InferenceRequest& request,
                        const Config& config);
std::string completion_json(const std::string& content, const InferenceStats& stats,
                            bool is_chat);
size_t complete_utf8_prefix_length(const std::string& text);

} // namespace radixforge::openai_api
