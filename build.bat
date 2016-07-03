@echo off
cl /EHsc /nologo main.cpp src/base.cpp src/fund.cpp /Iinc/ /I%appdata%/EPiC/inc /Febin/cuben.exe
del *.obj
