@echo off
cd /d "%~dp0"
echo Checking virtual environment and sentence_transformers import...
.\.venv\Scripts\python.exe -c "import sys; print('Python:', sys.executable); import sentence_transformers; print('sentence_transformers import OK')"
if errorlevel 1 (
    echo.
    echo ERROR: sentence_transformers import failed in the venv. Install requirements with:
    echo    .\.venv\Scripts\python.exe -m pip install -r requirements.txt
    pause
    goto :EOF
)
echo Starting Streamlit with the project venv...
.\.venv\Scripts\python.exe -m streamlit run app.py

