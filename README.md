# cuben

Computational C/C++ numerics library, structured around Timothy Sauer's "Numerical Analysis"

## Original Project

https://code.google.com/archive/p/cuben/

## Background

The “C/C++ Computational Numerics” library is an open-source (MIT license) set of methods and classes for solving both common and complex problems in numerical mathematics. It is designed and written in C/C++ (leveraging the Eigen library for vector and matrix support) for fast execution and near-universal platform support. Problems include multiple root-finding algorithms, numerous methods for solving systems of equations, classes for modeling different categories of numerical functions, ODE and PDE solvers, pseudo-random number generation, least-squares tools, and many others.

## Project Information

The project was created on Apr 30, 2013.

* License: MIT License

* hg-based source control

* Labels:

  - CPlusPlus

  - numerics
  
  - Mathematics

## Getting Started

Once cloned, ensure the submodules are present:

```sh
> git submodule update --init --recursive
```

Then, you can build the static libraries in the usual CMake two-step:

```sh
> cmake -S . -B build
> cmake --build build
```

Once everything is built you can also use CTest to run the unit tests from the command line:

```sh
> ctest --test-dir build -C Debug
```
