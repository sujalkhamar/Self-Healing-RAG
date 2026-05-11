@echo off
SETLOCAL EnableDelayedExpansion

:: --- CONFIGURATION ---
set VENV_DIR=venv
set REQS_FILE=requirements.txt
set MAIN_SCRIPT=main.py
set ENV_FILE=.env
set ENV_EXAMPLE=.env.example

echo ==============================================================
echo        Self-Healing RAG: Setup and Launch Script
echo ==============================================================
echo.

:: 1. Check for Python
python --version >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Python is not installed or not in your PATH.
    echo Please install Python from https://www.python.org/
    pause
    exit /b 1
)

:: 2. Create Virtual Environment if it doesn't exist
if not exist %VENV_DIR% (
    echo [INFO] Creating virtual environment in .\%VENV_DIR%...
    python -m venv %VENV_DIR%
    if %ERRORLEVEL% neq 0 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
    echo [SUCCESS] Virtual environment created.
)

:: 3. Activate Virtual Environment
echo [INFO] Activating virtual environment...
call %VENV_DIR%\Scripts\activate
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Failed to activate virtual environment.
    pause
    exit /b 1
)

:: 4. Install/Update Dependencies
echo [INFO] Checking/Installing dependencies from %REQS_FILE%...
pip install -r %REQS_FILE%
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Failed to install dependencies.
    pause
    exit /b 1
)

:: 5. Handle .env file
if not exist %ENV_FILE% (
    if exist %ENV_EXAMPLE% (
        echo [WARN] %ENV_FILE% not found. Copying from %ENV_EXAMPLE%...
        copy %ENV_EXAMPLE% %ENV_FILE%
        echo [IMPORTANT] Please edit %ENV_FILE% and add your API keys!
    ) else (
        echo [WARN] Neither %ENV_FILE% nor %ENV_EXAMPLE% found. 
        echo Skipping environment file setup.
    )
)

:: 6. Run the Application
echo.
echo [INFO] Starting the Self-Healing RAG Pipeline...
echo --------------------------------------------------------------
:: Set tokenizer parallelism to false to avoid warnings/hangs on Windows
set TOKENIZERS_PARALLELISM=false

:: Run main.py and pass all arguments from the .bat call
python %MAIN_SCRIPT% %*

if %ERRORLEVEL% neq 0 (
    echo.
    echo [ERROR] Application exited with error code %ERRORLEVEL%
    pause
)

echo.
echo [INFO] Script execution finished.
pause
ENDLOCAL
