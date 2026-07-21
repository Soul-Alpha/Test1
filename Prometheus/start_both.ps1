# Prometheus — Start both dashboards
#
# Streamlit  →  http://localhost:8501
# Gradio     →  http://localhost:7860
#
# Usage:
#   Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
#   .\start_both.ps1
#
# Optional flags:
#   .\start_both.ps1 -GradioShare      # create public Gradio URL via ngrok
#   .\start_both.ps1 -StreamlitOnly    # skip Gradio
#   .\start_both.ps1 -GradioOnly       # skip Streamlit

param(
    [switch]$GradioShare,
    [switch]$StreamlitOnly,
    [switch]$GradioOnly
)

$VENV    = "C:\Users\Chaba\Documents\tradingBots\.venv\Scripts"
$DIR     = "C:\Users\Chaba\Documents\tradingBots\Prometheus"
$PYTHON  = "$VENV\python.exe"
$STREAMLIT = "$VENV\streamlit.exe"

Write-Host ""
Write-Host "  Prometheus Dashboard Launcher" -ForegroundColor Cyan
Write-Host "  ─────────────────────────────────────" -ForegroundColor DarkGray

# ── Streamlit ──────────────────────────────────────────────────────────────────
if (-not $GradioOnly) {
    Write-Host "  Starting Streamlit  →  http://localhost:8501" -ForegroundColor Green
    $stArgs = @(
        "run", "$DIR\ui\prometheus_command_center.py",
        "--server.headless", "true",
        "--server.port", "8501",
        "--server.fileWatcherType", "none"
    )
    $stJob = Start-Job -Name "Streamlit" -ScriptBlock {
        param($exe, $args_)
        & $exe @args_
    } -ArgumentList $STREAMLIT, $stArgs
}

# ── Gradio ─────────────────────────────────────────────────────────────────────
if (-not $StreamlitOnly) {
    $gradioShare = if ($GradioShare) { "--share" } else { "" }
    Write-Host "  Starting Gradio     →  http://localhost:7860" -ForegroundColor Magenta
    $grArgs = @(
        "$DIR\ui\gradio_dashboard.py",
        "--port", "7860",
        "--host", "0.0.0.0"
    )
    if ($GradioShare) { $grArgs += "--share" }
    $grJob = Start-Job -Name "Gradio" -ScriptBlock {
        param($exe, $dir_, $args_)
        Set-Location $dir_
        & $exe @args_
    } -ArgumentList $PYTHON, $DIR, $grArgs
}

Write-Host ""
Write-Host "  Both processes running as background jobs." -ForegroundColor DarkGray
Write-Host "  Press  Ctrl+C  to stop this script." -ForegroundColor DarkGray
Write-Host "  Use    Get-Job | Stop-Job  to kill the dashboards." -ForegroundColor DarkGray
Write-Host ""

# ── Stream output until Ctrl+C ─────────────────────────────────────────────────
try {
    while ($true) {
        Start-Sleep -Seconds 5

        if (-not $GradioOnly -and $stJob) {
            $stOut = Receive-Job -Job $stJob 2>&1
            if ($stOut) { $stOut | ForEach-Object { Write-Host "[Streamlit] $_" -ForegroundColor Green } }
        }

        if (-not $StreamlitOnly -and $grJob) {
            $grOut = Receive-Job -Job $grJob 2>&1
            if ($grOut) { $grOut | ForEach-Object { Write-Host "[Gradio]    $_" -ForegroundColor Magenta } }
        }
    }
}
finally {
    Write-Host "`n  Stopping dashboards..." -ForegroundColor Yellow
    if ($stJob) { Stop-Job $stJob; Remove-Job $stJob }
    if ($grJob) { Stop-Job $grJob; Remove-Job $grJob }
    Write-Host "  Done." -ForegroundColor DarkGray
}
