@echo off
set ver=0.0.1
set name=cuben
cl /EHsc /nologo main.cpp src/base.cpp src/fund.cpp src/equations.cpp src/systems.cpp src/interp.cpp /Iinc/ /I%appdata%/EPiC/inc /Febin/%name%.exe
lib base.obj fund.obj equations.obj systems.obj interp.obj -OUT:lib/%name%.lib
copy bin\%name%.exe bin\%name%-%ver%.exe
copy lib\%name%.lib lib\%name%-%ver%.lib
