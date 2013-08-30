@echo off
g++ driver.cpp src/Base.cpp src/Fund.cpp src/Equations.cpp src/Systems.cpp src/Interp.cpp src/LeastSq.cpp src/DiffInt.cpp src/Ode.cpp src/Bvp.cpp src/Pde.cpp src/Rand.cpp src/Trig.cpp src/Compress.cpp -I C:/Libraries/ -o bin/cuben.exe
