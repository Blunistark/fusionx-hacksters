@echo off
echo =========================================
echo Launching FusionX Engine Layer
echo FastAPI running on http://localhost:8000
echo =========================================

:: Activate the virtual environment
call .venv\Scripts\activate.bat

:: Navigate to the engine module
cd engine

:: Run the FastAPI server
python main.py

echo.
pause
