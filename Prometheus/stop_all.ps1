# =============================================================================
# Prometheus — Full Stack Shutdown Script
# Stops: Prometheus bot, Zeus dashboard, Hermes bot/dashboard, Hermes Academy dashboard,
#        Prometheus dashboard
#        and kills all Cloudflare tunnel processes.
# Usage: powershell -ExecutionPolicy Bypass -File stop_all.ps1
# =============================================================================

$ErrorActionPreference = "Continue"
$ROOT       = Split-Path -Parent $MyInvocation.MyCommand.Definition
$statusFile = Join-Path $ROOT "live_bot\start_all_pids.json"

Write-Host "Stopping all Prometheus services..." -ForegroundColor Cyan

# ── 1. Kill by saved PID file ────────────────────────────────────────────────
if (Test-Path $statusFile) {
    $saved = Get-Content $statusFile | ConvertFrom-Json
    $saved.PSObject.Properties | ForEach-Object {
        $val = $_.Value
        if ($val -match '^\d+$') {
            $procId = [int]$val
            try {
                Stop-Process -Id $procId -Force -ErrorAction SilentlyContinue
                Write-Host "Stopped PID $procId ($($_.Name))" -ForegroundColor Green
            } catch {}
        }
    }
    Remove-Item $statusFile -Force -ErrorAction SilentlyContinue
}

# ── 2. Kill all cloudflared tunnel processes ─────────────────────────────────
Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like "*cloudflared*tunnel*" } | ForEach-Object {
    Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
    Write-Host "Stopped cloudflared PID $($_.ProcessId)" -ForegroundColor Green
}

# ── 3. Kill streamlit dashboard processes on 8501/8502/8503/8504/8505/8506/8507/8508/8509/8510/8511 ─────────
Get-CimInstance Win32_Process | Where-Object {
    $_.CommandLine -like "*streamlit*" -and (
        $_.CommandLine -like "*8501*" -or
        $_.CommandLine -like "*8502*" -or
        $_.CommandLine -like "*8503*" -or
        $_.CommandLine -like "*8504*" -or
        $_.CommandLine -like "*8505*" -or
        $_.CommandLine -like "*8506*" -or
        $_.CommandLine -like "*8507*" -or
        $_.CommandLine -like "*8508*" -or
        $_.CommandLine -like "*8509*" -or
        $_.CommandLine -like "*8510*" -or
        $_.CommandLine -like "*8511*" -or
        $_.CommandLine -like "*dashboard*" -or
        $_.CommandLine -like "*hermes_dashboard*" -or
        $_.CommandLine -like "*hermes_academy_dashboard*" -or
        $_.CommandLine -like "*hermes_return_dashboard*" -or
        $_.CommandLine -like "*hermes_pattern_context_dashboard*" -or
        $_.CommandLine -like "*olympus_observability_dashboard*" -or
        $_.CommandLine -like "*prometheus_execution_academy_dashboard*" -or
        $_.CommandLine -like "*prometheus_academy_observability_dashboard*" -or
        $_.CommandLine -like "*prometheus_evolution_dashboard*" -or
        $_.CommandLine -like "*knowledge_growth_dashboard*" -or
        $_.CommandLine -like "*zeus_dashboard*"
    )
} | ForEach-Object {
    Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
    Write-Host "Stopped streamlit PID $($_.ProcessId) (port match)" -ForegroundColor Green
}

# ── 4. Kill Prometheus trader.py ─────────────────────────────────────────────
Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like "*trader.py*" } | ForEach-Object {
    Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
    Write-Host "Stopped trader.py PID $($_.ProcessId)" -ForegroundColor Green
}

# ── 5. Kill Hermes bot ────────────────────────────────────────────────────────
Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like "*hermes*" -and $_.CommandLine -like "*python*" } | ForEach-Object {
    Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
    Write-Host "Stopped Hermes PID $($_.ProcessId)" -ForegroundColor Green
}

Write-Host "`nAll services stopped." -ForegroundColor Green
