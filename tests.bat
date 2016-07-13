@echo off
for %%f in (test\*.cpp) do (
	cl /EHsc /nologo /Iinc /I%appdata%/EPiC/inc lib/cuben.lib test\%%~nf.cpp /Fetest\%%~nf.exe
)
