function(radixforge_add_fuzzer name source)
    add_executable(${name} ${source} ../src/openai_api.cpp)
    target_include_directories(${name} PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/vendor/llama.cpp/include
        ${CMAKE_SOURCE_DIR}/vendor/llama.cpp/ggml/include
    )
    target_compile_options(${name} PRIVATE -fsanitize=fuzzer,address,undefined)
    target_link_options(${name} PRIVATE -fsanitize=fuzzer,address,undefined)
    radixforge_configure_target(${name})
endfunction()

radixforge_add_fuzzer(fuzz_json fuzz_json.cpp)
radixforge_add_fuzzer(fuzz_openai_request fuzz_openai_request.cpp)
