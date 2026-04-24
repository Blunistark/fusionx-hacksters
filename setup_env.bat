@echo off
echo =========================================
echo Setting up FusionX Virtual Environment...
echo =========================================

:: Check if .venv already exists
if exist ".venv\" (
    echo [INFO] Virtual environment '.venv' already exists.
) else (
    echo [INFO] Creating virtual environment '.venv'...
    python -m venv .venv
)

:: Activate the virtual environment
echo [INFO] Activating virtual environment...
call .venv\Scripts\activate.bat

:: Install dependencies
echo [INFO] Upgrading pip...
python -m pip install --upgrade pip

echo [INFO] Installing Perception Layer requirements...
pip install -r perception\requirements.txt

echo =========================================
echo Setup Complete!
echo To activate the environment in the future, run:
echo .venv\Scripts\activate
echo =========================================
cmd /k
