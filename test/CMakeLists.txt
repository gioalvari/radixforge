enable_testing()

add_executable(test_radix_tree
    test_radix_tree.cpp
    ../src/radix_tree.cpp
)

target_include_directories(test_radix_tree PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/vendor/llama.cpp/include
)

target_link_libraries(test_radix_tree PRIVATE llama)

if(APPLE AND RADIXFORGE_METAL)
    target_link_libraries(test_radix_tree PRIVATE
        "-framework Metal"
        "-framework Foundation"
    )
endif()

add_test(NAME radix_tree_unit_tests COMMAND test_radix_tree)

add_executable(test_kv_mapper
    test_kv_mapper.cpp
    ../src/kv_mapper.cpp
    ../src/radix_tree.cpp
    ../src/llama_bridge.cpp
)

target_include_directories(test_kv_mapper PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/vendor/llama.cpp/include
)

target_link_libraries(test_kv_mapper PRIVATE llama)

if(APPLE AND RADIXFORGE_METAL)
    target_link_libraries(test_kv_mapper PRIVATE
        "-framework Metal"
        "-framework Foundation"
    )
endif()

add_test(NAME kv_mapper_unit_tests COMMAND test_kv_mapper)
