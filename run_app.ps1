Set-Location -Path "$PSScriptRoot"
Write-Host "Checking virtual environment and sentence_transformers import..."
& ".\.venv\Scripts\python.exe" -c "import sys; print('Python:', sys.executable); import sentence_transformers; print('sentence_transformers import OK')"
if ($LASTEXITCODE -ne 0) {
    Write-Error "ERROR: sentence_transformers import failed in the venv. Install requirements with:`n    .\.venv\Scripts\python.exe -m pip install -r requirements.txt"
    exit 1
}
Write-Host "Starting Streamlit with the project venv..."
& ".\.venv\Scripts\python.exe" -m streamlit run app.py
