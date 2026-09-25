cmake_minimum_required(VERSION 3.21)
project(radixforge VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# --- Options ---
option(RADIXFORGE_METAL "Enable Metal acceleration (macOS)" ON)
option(RADIXFORGE_BUILD_TESTS "Build unit tests" ON)
option(RADIXFORGE_BUILD_TOOLS "Build standalone feasibility tools" ON)

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
        "Run: git clone https://github.com/ggerganov/llama.cpp vendor/llama.cpp")
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

target_compile_options(radixforge PRIVATE -Wall -Wextra -Wpedantic)

if(RADIXFORGE_BUILD_TOOLS)
    add_executable(chunk_spike tools/chunk_spike.cpp)
    set_target_properties(chunk_spike PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tools"
    )
    target_include_directories(chunk_spike PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/vendor/llama.cpp/include
    )
    target_link_libraries(chunk_spike PRIVATE llama ggml)
    target_compile_options(chunk_spike PRIVATE -Wall -Wextra -Wpedantic)
    if(APPLE AND RADIXFORGE_METAL)
        target_link_libraries(chunk_spike PRIVATE "-framework Metal" "-framework Foundation")
    endif()
endif()

if(RADIXFORGE_BUILD_TESTS)
    enable_testing()
    add_subdirectory(test)
endif()
