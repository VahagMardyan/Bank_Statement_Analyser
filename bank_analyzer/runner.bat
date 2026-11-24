@echo off
rem Starts the Bank Statement Analyser with the project's .venv.
rem Works from any folder and from a double-click.
cd /d "%~dp0"

if not exist ".venv\Scripts\activate.bat" (
    echo .venv not found. Create it first:
    echo     python -m venv .venv
    echo     .venv\Scripts\activate
    echo     pip install -r ..\requirements.txt
    pause
    exit /b 1
)

call ".venv\Scripts\activate.bat"
streamlit run app.py

rem Keep the window open if Streamlit stopped with an error
if errorlevel 1 pause
