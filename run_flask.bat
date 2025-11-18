@echo off
REM Simple script to run the Flask application on Windows

echo Starting RAG Pipeline Flask Server...
echo.

REM Check if .env file exists
if not exist .env (
    echo Warning: .env file not found!
    echo Creating .env from .env.example...
    if exist .env.example (
        copy .env.example .env
        echo Please edit .env file with your API keys before running queries.
    ) else (
        echo Please create a .env file with your configuration.
    )
    echo.
)

REM Check if virtual environment is activated
if "%VIRTUAL_ENV%"=="" (
    echo Warning: No virtual environment detected.
    echo Consider activating a virtual environment first:
    echo   python -m venv venv
    echo   venv\Scripts\activate
    echo.
)

REM Run the Flask application
python app.py %*
