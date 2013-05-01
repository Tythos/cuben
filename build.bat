@echo off
g++ driver.cpp src/Base.cpp src/Fund.cpp src/Equations.cpp src/Systems.cpp src/Interp.cpp src/LeastSq.cpp src/DiffInt.cpp src/Ode.cpp -I C:/Libraries/ -o bin/cuben.exe
