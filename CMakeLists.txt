cmake_minimum_required(VERSION 3.10.2)
project(cuben)
add_library(cuben STATIC
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
add_executable(diffint test_diffint.cpp)
add_executable(equations test_equations.cpp)
add_executable(fund test_fund.cpp)
add_executable(interp test_interp.cpp)
add_executable(leastsq test_leastsq.cpp)
add_executable(ode test_ode.cpp)
add_executable(systems test_systems.cpp)
