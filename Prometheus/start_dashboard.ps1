# Prometheus Dashboard Watchdog
# Keeps the Streamlit dashboard alive — auto-restarts if it crashes.
# Usage: right-click -> Run with PowerShell  (or pin it to startup)

$PYTHON = "c:\Users\Chaba\Documents\tradingBots\.venv\Scripts\python.exe"
$SCRIPT = "ui\prometheus_command_center.py"
$DIR    = "c:\Users\Chaba\Documents\tradingBots\Prometheus"
$PORT   = 8501
$ERRLOG = "$DIR\live_bot\dashboard_err.log"

Write-Host "Prometheus Dashboard Watchdog started (port $PORT)" -ForegroundColor Cyan

while ($true) {
    Write-Host "$(Get-Date -Format 'HH:mm:ss') Starting dashboard..." -ForegroundColor Green
    $proc = Start-Process -FilePath $PYTHON `
        -ArgumentList "-m", "streamlit", "run", $SCRIPT, "--server.headless=true", "--server.port=$PORT", "--server.runOnSave=false", "--server.address=0.0.0.0" `
        -WorkingDirectory $DIR `
        -RedirectStandardError $ERRLOG `
        -WindowStyle Minimized `
        -PassThru
    $proc.WaitForExit()
    $exit = $proc.ExitCode
    Write-Host "$(Get-Date -Format 'HH:mm:ss') Dashboard exited (code $exit). Restarting in 5s..." -ForegroundColor Yellow
    Start-Sleep -Seconds 5
}
