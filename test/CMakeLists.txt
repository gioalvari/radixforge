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
