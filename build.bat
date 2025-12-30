@echo off
REM Build script for Interactive Torque standalone application
REM This script builds a standalone Windows executable using PyInstaller

echo ========================================
echo Building Interactive Torque Application
echo ========================================
echo.

REM Check if virtual environment is activated
if not defined VIRTUAL_ENV (
    echo Warning: Virtual environment not detected.
    echo Please activate your virtual environment first with:
    echo   .\venv\Scripts\Activate.ps1
    echo.
    pause
    exit /b 1
)

echo Installing/updating PyInstaller...
pip install --upgrade pyinstaller pywebview
if errorlevel 1 (
    echo Failed to install PyInstaller
    pause
    exit /b 1
)

echo.
echo Building standalone executable...
pyinstaller build_standalone.spec --clean
if errorlevel 1 (
    echo Build failed!
    pause
    exit /b 1
)

echo.
echo ========================================
echo Build completed successfully!
echo ========================================
echo.
echo The standalone application is located in:
echo   dist\InteractiveTorqueApp\
echo.
echo To run the application:
echo   cd dist\InteractiveTorqueApp
echo   InteractiveTorqueApp.exe
echo.
pause
