cmake_minimum_required(VERSION 3.10.2)
project(cuben)

# define dependencies
add_subdirectory(eigen EXCLUDE_FROM_ALL)
target_link_libraries(${PROJECT_NAME} PRIVATE Eigen3::Eigen)

# define library contents
add_library(${PROJECT_NAME} STATIC
    Base.cpp
    Bvp.cpp
    Compress.cpp
    DiffInt.cpp
    Equations.cpp
    Fund.cpp
    Interp.cpp
    LeastSq.cpp
    Ode.cpp
    Pde.cpp
    Rand.cpp
    Systems.cpp
    Trig.cpp
)

# define test executables
enable_testing()
find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIR})
add_test(diffint tests/test_diffint.cpp)
