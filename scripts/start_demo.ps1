# start_demo.ps1 — One-click FYP demo launcher
Write-Host "🚀 Starting WiM-RAG Demo..." -ForegroundColor Cyan

# 1. Start llama.cpp LLM server (background)
Write-Host "Starting local LLM server..." -ForegroundColor Yellow
$llm = Start-Process -FilePath ".\models\llama-b8368-bin-win-cuda-12.4-x64\llama-server.exe" `
    -ArgumentList "-m", ".\models\Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf", `
                  "-ngl", "99", "-c", "8192", "--host", "0.0.0.0", "--port", "8080" `
    -PassThru -WindowStyle Minimized
Start-Sleep -Seconds 10

# 2. Activate venv and start FastAPI backend
Write-Host "Starting FastAPI backend..." -ForegroundColor Yellow
$backend = Start-Process -FilePath ".\venv\Scripts\python.exe" `
    -ArgumentList "fastapi_server.py" `
    -PassThru -WindowStyle Minimized
Start-Sleep -Seconds 15

# 3. Start Streamlit frontend
Write-Host "Starting Streamlit UI..." -ForegroundColor Yellow
Start-Process -FilePath ".\venv\Scripts\streamlit.exe" `
    -ArgumentList "run", "app.py", "--server.port", "8501"

Write-Host "`n✅ Demo ready!" -ForegroundColor Green
Write-Host "   Frontend: http://localhost:8501" -ForegroundColor White
Write-Host "   Backend:  http://localhost:8000" -ForegroundColor White
Write-Host "   LLM:      http://localhost:8080" -ForegroundColor White
