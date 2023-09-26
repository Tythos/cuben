cmake_minimum_required(VERSION 3.10.2)
project(cuben)

# include dependencies
add_subdirectory(eigen EXCLUDE_FROM_ALL)
add_subdirectory(googletest EXCLUDE_FROM_ALL)

# define library contents
add_library(${PROJECT_NAME} STATIC
    constants.cpp
    # Base.cpp
    # Bvp.cpp
    # Compress.cpp
    # DiffInt.cpp
    # Equations.cpp
    # Fund.cpp
    # Interp.cpp
    # LeastSq.cpp
    # Ode.cpp
    # Pde.cpp
    # Rand.cpp
    # Systems.cpp
    # Trig.cpp
)

# link against dependencies
target_link_libraries(${PROJECT_NAME} PRIVATE Eigen3::Eigen)

# # define test executables
# enable_testing()
# add_executable(diffint tests/test_diffint.cpp)
# target_link_libraries(diffint gtest gtest_main)
# add_test(diffint diffint)
