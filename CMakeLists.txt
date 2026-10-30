cmake_minimum_required(VERSION 3.21)
project(radixforge VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# --- Options ---
option(RADIXFORGE_METAL "Enable Metal acceleration (macOS)" ON)
option(RADIXFORGE_BUILD_TESTS "Build unit tests" ON)
option(RADIXFORGE_BUILD_TOOLS "Build standalone feasibility tools" ON)
option(RADIXFORGE_UNIT_TESTS_ONLY "Build only unit-test targets without llama.cpp" OFF)
option(RADIXFORGE_WERROR "Treat RadixForge warnings as errors" OFF)
set(RADIXFORGE_SANITIZE "" CACHE STRING
    "Sanitizer for RadixForge targets: empty, address,undefined, or thread")
set_property(CACHE RADIXFORGE_SANITIZE PROPERTY STRINGS "" "address,undefined" "thread")

if(RADIXFORGE_SANITIZE STREQUAL "address,undefined")
    set(RADIXFORGE_SANITIZER_FLAG "-fsanitize=address,undefined")
elseif(RADIXFORGE_SANITIZE STREQUAL "thread")
    set(RADIXFORGE_SANITIZER_FLAG "-fsanitize=thread")
elseif(NOT RADIXFORGE_SANITIZE STREQUAL "")
    message(FATAL_ERROR
        "RADIXFORGE_SANITIZE must be empty, address,undefined, or thread; got '${RADIXFORGE_SANITIZE}'")
endif()

function(radixforge_configure_target target)
    target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic)
    if(RADIXFORGE_WERROR)
        target_compile_options(${target} PRIVATE -Werror)
    endif()
    if(RADIXFORGE_SANITIZER_FLAG)
        target_compile_options(${target} PRIVATE ${RADIXFORGE_SANITIZER_FLAG})
        target_link_options(${target} PRIVATE ${RADIXFORGE_SANITIZER_FLAG})
    endif()
endfunction()

if(NOT RADIXFORGE_UNIT_TESTS_ONLY)
    # --- llama.cpp as subdirectory ---
    # Disable llama.cpp extras we don't need
    set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
    set(GGML_METAL ${RADIXFORGE_METAL} CACHE BOOL "" FORCE)
    set(GGML_ACCELERATE ON CACHE BOOL "" FORCE)

    if(EXISTS "${CMAKE_SOURCE_DIR}/vendor/llama.cpp/CMakeLists.txt")
        add_subdirectory(vendor/llama.cpp)
    else()
        message(FATAL_ERROR
            "llama.cpp not found in vendor/llama.cpp.\n"
            "Run: git submodule update --init --recursive")
    endif()

    # --- RadixForge executable ---
    add_executable(radixforge
        src/main.cpp
        src/llama_bridge.cpp
        src/radix_tree.cpp
        src/kv_mapper.cpp
        src/server.cpp
        vendor/llama.cpp/vendor/cpp-httplib/httplib.cpp
    )

    target_include_directories(radixforge PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/vendor/llama.cpp/include
        ${CMAKE_SOURCE_DIR}/vendor/llama.cpp/ggml/include
        ${CMAKE_SOURCE_DIR}/vendor/llama.cpp/vendor/cpp-httplib
    )

    # Disable httplib OpenSSL and zlib — we only need plain HTTP
    target_compile_definitions(radixforge PRIVATE
        CPPHTTPLIB_NO_EXCEPTIONS=1
    )

    target_link_libraries(radixforge PRIVATE llama ggml)

    # LTO for release builds
    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        set_target_properties(radixforge PROPERTIES INTERPROCEDURAL_OPTIMIZATION ON)
    endif()

    # macOS: link Metal and Accelerate frameworks
    if(APPLE AND RADIXFORGE_METAL)
        target_link_libraries(radixforge PRIVATE "-framework Metal" "-framework Foundation")
    endif()

    radixforge_configure_target(radixforge)

    if(RADIXFORGE_BUILD_TOOLS)
        add_executable(chunk_spike tools/chunk_spike.cpp)
        set_target_properties(chunk_spike PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tools"
        )
        target_include_directories(chunk_spike PRIVATE
            ${CMAKE_SOURCE_DIR}/include
            ${CMAKE_SOURCE_DIR}/vendor/llama.cpp/include
            ${CMAKE_SOURCE_DIR}/vendor/llama.cpp/ggml/include
        )
        target_link_libraries(chunk_spike PRIVATE llama ggml)
        radixforge_configure_target(chunk_spike)
        if(APPLE AND RADIXFORGE_METAL)
            target_link_libraries(chunk_spike PRIVATE "-framework Metal" "-framework Foundation")
        endif()

        add_executable(moe_trace tools/moe_trace.cpp)
        set_target_properties(moe_trace PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tools"
        )
        target_include_directories(moe_trace PRIVATE
            ${CMAKE_SOURCE_DIR}/include
            ${CMAKE_SOURCE_DIR}/vendor/llama.cpp/include
            ${CMAKE_SOURCE_DIR}/vendor/llama.cpp/ggml/include
        )
        target_link_libraries(moe_trace PRIVATE llama ggml ggml-base)
        radixforge_configure_target(moe_trace)
    endif()
endif()

if(RADIXFORGE_BUILD_TESTS)
    enable_testing()
    add_subdirectory(test)
endif()
