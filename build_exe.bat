@echo off
echo Building UAD Pro Console executable...
echo.

REM Install PyInstaller if not present
pip install pyinstaller

REM Build executable
pyinstaller --onefile --windowed --name "UAD_Pro_Console" uad_console.py

echo.
echo Build complete! Executable is in dist\UAD_Pro_Console.exe
pause
