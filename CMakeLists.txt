# top-level project build configuration
cmake_minimum_required(VERSION 3.14)
project(cuben)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# one-time test configuraiton
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
add_subdirectory("googletest/" EXCLUDE_FROM_ALL)
add_subdirectory("eigen/" EXCLUDE_FROM_ALL)
enable_testing()

# define primary build target
add_library(${PROJECT_NAME} STATIC
    "constants.cpp"
    "exceptions.cpp"
    "Polynomial.cpp"
    "fundamentals.cpp"
    "equations.cpp"
    "systems.cpp"
    "interp.cpp"
    "CubicSpline.cpp"
    "BezierSpline.cpp"
    "leastsq.cpp"
    "diffint.cpp"
    "ode.cpp"
    "bvp.cpp"
    "pde.cpp"
    "rand.cpp"
    "Prng.cpp"
    "Lcg.cpp"
    "MinStd.cpp"
    "Randu.cpp"
    "Alfg.cpp"
    "Mlfg.cpp"
    "MTwist.cpp"
    "Bbs.cpp"
    "Norm.cpp"
    "Halton.cpp"
    "RandomWalk.cpp"
    "RandomEscape.cpp"
    "Brownian.cpp"
    "BrownianBridge.cpp"
    "BlackScholes.cpp"
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
add_gtest(test_constants "tests/test_constants.cpp")
add_gtest(test_exceptions "tests/test_exceptions.cpp")
add_gtest(test_Polynomial "tests/test_Polynomial.cpp")
add_gtest(test_fundamentals "tests/test_fundamentals.cpp")
add_gtest(test_equations "tests/test_equations.cpp")
add_gtest(test_systems "tests/test_systems.cpp")
add_gtest(test_interp "tests/test_interp.cpp")
add_gtest(test_CubicSpline "tests/test_CubicSpline.cpp")
add_gtest(test_BezierSpline "tests/test_BezierSpline.cpp")
add_gtest(test_leastsq "tests/test_leastsq.cpp")
add_gtest(test_diffint "tests/test_diffint.cpp")
add_gtest(test_ode "tests/test_ode.cpp")
add_gtest(test_bvp "tests/test_bvp.cpp")
add_gtest(test_pde "tests/test_pde.cpp")
add_gtest(test_rand "tests/test_rand.cpp")
add_gtest(test_Prng "tests/test_Prng.cpp")
add_gtest(test_Lcg "tests/test_Lcg.cpp")
add_gtest(test_MinStd "tests/test_MinStd.cpp")
add_gtest(test_Randu "tests/test_Randu.cpp")
add_gtest(test_Alfg "tests/test_Alfg.cpp")
add_gtest(test_Mlfg "tests/test_Mlfg.cpp")
add_gtest(test_MTwist "tests/test_MTwist.cpp")
add_gtest(test_Bbs "tests/test_Bbs.cpp")
add_gtest(test_Norm "tests/test_Norm.cpp")
add_gtest(test_Halton "tests/test_Halton.cpp")
add_gtest(test_RandomWalk "tests/test_RandomWalk.cpp")
add_gtest(test_RandomEscape "tests/test_RandomEscape.cpp")
add_gtest(test_Brownian "tests/test_Brownian.cpp")
add_gtest(test_BrownianBridge "tests/test_BrownianBridge.cpp")
add_gtest(test_BlackScholes "tests/test_BlackScholes.cpp")
