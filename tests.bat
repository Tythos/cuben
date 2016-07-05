@echo off
cl /EHsc /nologo /Iinc /I%appdata%/EPiC/inc test\test_systems.cpp src\systems.cpp src\equations.cpp src\base.cpp src\fund.cpp
