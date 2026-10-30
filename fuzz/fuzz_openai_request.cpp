#include "radixforge/openai_api.h"

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <string>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    radixforge::Config config;
    radixforge::InferenceRequest request;
    try {
        radixforge::openai_api::parse_request_body(
            std::string(reinterpret_cast<const char*>(data), size), request, config);
    } catch (const std::exception&) {
    }
    return 0;
}

#ifdef RADIXFORGE_FUZZ_STANDALONE
int main(int argc, char** argv) {
    for (int index = 1; index < argc; ++index) {
        std::ifstream file(argv[index], std::ios::binary);
        const std::string input((std::istreambuf_iterator<char>(file)), {});
        LLVMFuzzerTestOneInput(reinterpret_cast<const uint8_t*>(input.data()), input.size());
    }
    return 0;
}
#endif
