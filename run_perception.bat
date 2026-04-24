@echo off
echo =========================================
echo Launching FusionX Perception Layer
echo Running on CUDA (NVIDIA GPU Accelerated)
echo =========================================

:: Activate the virtual environment
call .venv\Scripts\activate.bat

:: Navigate to the perception module
cd perception

:: Run the tracker
python tracker.py

echo.
pause
