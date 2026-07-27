$ErrorActionPreference = "Stop"

if (Test-Path ".\.venv\Scripts\Activate.ps1") {
    .\.venv\Scripts\Activate.ps1
}

$env:API_URL = "http://127.0.0.1:8000"

streamlit run frontend/app.py `
    --server.address 127.0.0.1 `
    --server.port 8501
