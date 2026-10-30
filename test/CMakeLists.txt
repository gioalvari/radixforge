function(radixforge_add_unit_test name)
    add_executable(${name} ${ARGN})
    target_include_directories(${name} PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/vendor/llama.cpp/include
        ${CMAKE_SOURCE_DIR}/vendor/llama.cpp/ggml/include
    )
    radixforge_configure_target(${name})
    add_test(NAME ${name} COMMAND ${name})
endfunction()

radixforge_add_unit_test(test_radix_tree
    test_radix_tree.cpp
    ../src/radix_tree.cpp
)

# KVMapper is tested through a fake KvOps; no llama.cpp library is linked.
radixforge_add_unit_test(test_kv_mapper
    test_kv_mapper.cpp
    ../src/kv_mapper.cpp
    ../src/radix_tree.cpp
)

# OpenAI request parsing and JSON formatting are pure; this target does not link llama.cpp.
radixforge_add_unit_test(test_openai_api
    test_openai_api.cpp
    ../src/openai_api.cpp
)

function(radixforge_add_fuzz_replay_test name source corpus)
    add_executable(${name} ${source} ../src/openai_api.cpp)
    target_compile_definitions(${name} PRIVATE RADIXFORGE_FUZZ_STANDALONE=1)
    target_include_directories(${name} PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/vendor/llama.cpp/include
        ${CMAKE_SOURCE_DIR}/vendor/llama.cpp/ggml/include
    )
    radixforge_configure_target(${name})
    file(GLOB corpus_files CONFIGURE_DEPENDS "${CMAKE_SOURCE_DIR}/${corpus}/*")
    add_test(NAME ${name} COMMAND ${name} ${corpus_files})
endfunction()

# Replay the seed corpus without libFuzzer so every compiler validates fuzz inputs.
radixforge_add_fuzz_replay_test(test_fuzz_json_replay ../fuzz/fuzz_json.cpp fuzz/corpus)
radixforge_add_fuzz_replay_test(
    test_fuzz_openai_request_replay
    ../fuzz/fuzz_openai_request.cpp
    fuzz/corpus
)

# Keep unit/fuzz targets above the E2E blocks appended by parallel lanes.
if(NOT RADIXFORGE_UNIT_TESTS_ONLY AND EXISTS "${CMAKE_SOURCE_DIR}/test/test_e2e_determinism.py")
    set(RADIXFORGE_E2E_MODEL "$ENV{RADIXFORGE_E2E_MODEL}" CACHE FILEPATH
        "Path to a GGUF model for end-to-end tests")
    if(RADIXFORGE_E2E_MODEL)
        find_package(Python3 REQUIRED COMPONENTS Interpreter)
        add_test(
            NAME e2e_determinism
            COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_e2e_determinism.py
                --server $<TARGET_FILE:radixforge>
                --model ${RADIXFORGE_E2E_MODEL}
        )
        set_tests_properties(e2e_determinism PROPERTIES TIMEOUT 180)
    else()
        message(STATUS "Skipping e2e_determinism: set RADIXFORGE_E2E_MODEL to a .gguf path")
    endif()
endif()

if(NOT RADIXFORGE_UNIT_TESTS_ONLY AND EXISTS "${CMAKE_SOURCE_DIR}/test/test_e2e_robustness.py")
    set(RADIXFORGE_E2E_MODEL "$ENV{RADIXFORGE_E2E_MODEL}" CACHE FILEPATH
        "Path to a GGUF model for end-to-end tests")
    if(RADIXFORGE_E2E_MODEL)
        find_package(Python3 REQUIRED COMPONENTS Interpreter)
        add_test(
            NAME e2e_robustness
            COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_e2e_robustness.py
                --server $<TARGET_FILE:radixforge>
                --model ${RADIXFORGE_E2E_MODEL}
        )
        set_tests_properties(e2e_robustness PROPERTIES TIMEOUT 180)
    else()
        message(STATUS "Skipping e2e_robustness: set RADIXFORGE_E2E_MODEL to a .gguf path")
    endif()
endif()
