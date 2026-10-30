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

# Add KVMapper tests without linking llama.cpp, using its fake KV-ops adapter:
# radixforge_add_unit_test(test_kv_mapper test_kv_mapper.cpp ../src/kv_mapper.cpp ...)

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
