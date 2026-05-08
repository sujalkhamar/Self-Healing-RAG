@echo off
SETLOCAL
:: Disable tokenizer parallelism to prevent hangs on Windows
set TOKENIZERS_PARALLELISM=false

echo Starting Self-Healing RAG Pipeline...
echo.

:: Run the python script and pass any arguments provided to the bat file
python main.py %*

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Pipeline failed with exit code %ERRORLEVEL%
    pause
)

ENDLOCAL
