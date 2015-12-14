@echo off

set PYTHONEXE=C:\Python34\Python.exe

REM Retrieve corpus data if not present.
%PYTHONEXE% prepare.py

REM Run once to populate OS cache.
%PYTHONEXE% process_once.py

REM Timed process
echo "Using Python 2 with standard chunking:"
echo %time%
%PYTHONEXE% process_once.py
echo %time%

echo "Using Python 3 with standard chunking:"
echo %time%
%PYTHONEXE% process_once.py
echo %time%

echo "Using Python 2 with external chunking:"
echo %time%
%PYTHONEXE% process_once.py external
echo %time%

echo "Using Python 3 with external chunking:"
echo %time%
%PYTHONEXE% process_once.py external
echo %time%
