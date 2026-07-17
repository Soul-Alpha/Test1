# Zeus — Prometheus LTF Scalp Backtester Dashboard
# Runs on port 8502 (live-bot dashboard uses 8501)

$PY   = "C:\Users\Chaba\Documents\tradingBots\.venv\Scripts\python.exe"
$DASH = "C:\Users\Chaba\Documents\tradingBots\Prometheus\backtesting\zeus_dashboard.py"

Write-Host "Starting Zeus Dashboard on http://localhost:8502 ..." -ForegroundColor Cyan
& $PY -m streamlit run $DASH `
    --server.port=8502 `
    --server.headless=true `
    --server.address=0.0.0.0
