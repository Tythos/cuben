# top-level project build configuration
cmake_minimum_required(VERSION 3.14)
project(cuben)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# one-time test configuraiton
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
add_subdirectory(googletest EXCLUDE_FROM_ALL)
add_subdirectory(eigen EXCLUDE_FROM_ALL)
enable_testing()

# define primary build target
add_library(${PROJECT_NAME} STATIC
    constants.cpp
    exceptions.cpp
    polynomial.cpp
    fundamentals.cpp
)
target_link_libraries(${PROJECT_NAME} PRIVATE Eigen3::Eigen)

# define test hooks
function(add_gtest test_name test_source)
    add_executable(${test_name} ${test_source})
    target_include_directories(${test_name} PUBLIC ${CMAKE_SOURCE_DIR})
    target_link_libraries(${test_name} Eigen3::Eigen gtest gtest_main ${PROJECT_NAME})
    add_test(NAME ${test_name} COMMAND ${test_name})
endfunction()

# add, link, catalog each test; can be run once built with "> ctest --test-dir build -C Debug"
add_gtest(test_constants tests/test_constants.cpp)
add_gtest(test_exceptions tests/test_exceptions.cpp)
add_gtest(test_polynomial tests/test_polynomial.cpp)
add_gtest(test_fundamentals tests/test_fundamentals.cpp)
