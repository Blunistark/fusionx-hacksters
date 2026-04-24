@echo off
REM FusionX Streamlit Frontend Startup Script for Windows

echo.
echo ========================================
echo   FusionX Cricket Commentary System
echo   Streamlit Frontend Launcher
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.9+ from https://www.python.org
    pause
    exit /b 1
)

echo [✓] Python found
echo.

REM Navigate to UI directory
cd /d "%~dp0"
echo [✓] Working directory: %CD%

REM Check if root virtual environment exists
if not exist "..\.venv" (
    echo.
    echo ERROR: Root virtual environment not found in ..\.venv
    pause
    exit /b 1
)

REM Activate root virtual environment
echo.
echo [*] Activating root virtual environment...
call ..\.venv\Scripts\activate.bat
if errorlevel 1 (
    echo ERROR: Failed to activate virtual environment
    pause
    exit /b 1
)
echo [✓] Virtual environment activated

REM Check if requirements are installed
echo.
echo [*] Checking dependencies...
pip show streamlit >nul 2>&1
if errorlevel 1 (
    echo.
    echo [*] Installing dependencies from requirements.txt...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo ERROR: Failed to install dependencies
        pause
        exit /b 1
    )
    echo [✓] Dependencies installed
) else (
    echo [✓] Dependencies already installed
)

REM Check for .env file
echo.
if not exist ".env" (
    echo [!] WARNING: .env file not found
    echo [!] API keys may not be configured
    echo [!] Create a .env file with your API keys to enable LLM features
    echo.
    echo Example .env format:
    echo OPENAI_API_KEY=sk-...
    echo GOOGLE_API_KEY=...
    echo ANTHROPIC_API_KEY=...
    echo.
)

REM Launch Streamlit app
echo.
echo [*] Launching Streamlit application...
echo [*] The app will open in your default browser at http://localhost:8501
echo.
echo ========================================
echo   Press Ctrl+C to stop the server
echo ========================================
echo.

streamlit run app.py

REM If the user closes the browser or stops the app, show this message
echo.
echo [*] Streamlit server stopped
echo [*] Virtual environment is still active
echo [*] Type 'deactivate' to exit the virtual environment
echo.

pause
