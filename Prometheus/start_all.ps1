# =============================================================================
# Prometheus - Full Stack Startup Script
# Starts: Prometheus bot, Prometheus dashboard (8501), Zeus dashboard (8502),
#         Hermes bot, Hermes dashboard (8503), Hermes Academy dashboard (8504),
#         Hermes Return Intelligence dashboard (8505), Hermes Pattern Context dashboard (8506),
#         Olympus Observability dashboard (8507), Prometheus Execution Academy dashboard (8508),
#         Prometheus Academy Observability dashboard (8509), Prometheus Evolution dashboard (8510),
#         Olympus Command Center (8511), and optional tunnels.
# Usage: powershell -ExecutionPolicy Bypass -File start_all.ps1
# =============================================================================

param(
    [switch]$SkipBot,
    [switch]$SkipHermesBot,
    [switch]$SkipTunnels,
    [string]$LogDir = ""
)

$ErrorActionPreference = "Continue"
$ROOT = Split-Path -Parent $MyInvocation.MyCommand.Definition
$PYTHON = "c:\Users\Chaba\Documents\tradingBots\.venv\Scripts\python.exe"
$CF = "C:\Program Files (x86)\cloudflared\cloudflared.exe"
if ($LogDir -eq "") { $LogDir = Join-Path $ROOT "logs" }
New-Item -ItemType Directory -Path $LogDir -Force | Out-Null

function Start-Proc {
    param(
        [string]$Name,
        [string]$Exe,
        [string[]]$ProcParams,
        [string]$Cwd,
        [string]$Log
    )

    Write-Host "[$Name] Starting..." -ForegroundColor Cyan
    if ($Log) {
        $stdoutLog = $Log
        $stderrLog = [System.IO.Path]::ChangeExtension($Log, ".err.log")
        $proc = Start-Process -FilePath $Exe -ArgumentList $ProcParams -WorkingDirectory $Cwd -PassThru -WindowStyle Hidden -RedirectStandardOutput $stdoutLog -RedirectStandardError $stderrLog
    } else {
        $proc = Start-Process -FilePath $Exe -ArgumentList $ProcParams -WorkingDirectory $Cwd -PassThru -WindowStyle Hidden
    }
    Write-Host "[$Name] PID $($proc.Id)" -ForegroundColor Green
    return $proc.Id
}

function Start-Dashboard {
    param(
        [string]$Name,
        [string]$Script,
        [int]$Port,
        [string]$Log
    )

    $dashParams = @(
        "-m"
        "streamlit"
        "run"
        $Script
        "--server.port=$Port"
        "--server.headless=true"
        "--server.runOnSave=false"
        "--server.address=0.0.0.0"
    )

    return Start-Proc -Name $Name -Exe $PYTHON -ProcParams $dashParams -Cwd $ROOT -Log $Log
}

function Start-Tunnel {
    param(
        [int]$Port,
        [string]$Log
    )

    $tunnelParams = @(
        "tunnel"
        "--url"
        "http://127.0.0.1:$Port"
    )

    return Start-Proc -Name "CF-$Port" -Exe $CF -ProcParams $tunnelParams -Cwd $ROOT -Log $Log
}

function Test-PortFree {
    param([int]$Port)
    return -not (netstat -ano | findstr ":$Port" | findstr "LISTENING")
}

$statusFile = Join-Path $ROOT "live_bot\start_all_pids.json"
$pids = @{}

Write-Host "[Recovery] Reconstructing institutional state from persisted artifacts..." -ForegroundColor Cyan
try {
    & $PYTHON -m olympus.core.institutional_state_recovery --root $ROOT --write-snapshot | Out-Null
    Write-Host "[Recovery] Institutional recovery snapshot updated." -ForegroundColor Green
} catch {
    Write-Host "[Recovery] Recovery preflight failed; continuing with startup." -ForegroundColor Yellow
}

if (-not $SkipBot) {
    $botParams = @(
        "-m"
        "live_bot.trader"
        "--live"
        "--asset"
        "XAUUSDm"
        "--tf"
        "4H"
        "--min-grade"
        "B"
        "--min-score"
        "65"
        "--risk"
        "1.0"
        "--poll"
        "60"
        "--candles"
        "500"
        "--entry-mode"
        "market_any"
    )
    $pids["prometheus_bot"] = Start-Proc -Name "Prometheus-Bot" -Exe $PYTHON -ProcParams $botParams -Cwd $ROOT -Log (Join-Path $LogDir "prometheus_bot.log")
    Start-Sleep -Seconds 2
}

if (Test-PortFree 8501) {
    $pids["prometheus_dash"] = Start-Dashboard -Name "Prometheus-Dash" -Script "ui\dashboard.py" -Port 8501 -Log (Join-Path $LogDir "prometheus_dash.log")
    Start-Sleep -Seconds 3
} else {
    Write-Host "[Prometheus-Dash] Port 8501 already in use - skipping." -ForegroundColor Yellow
}

if (Test-PortFree 8502) {
    $pids["zeus_dash"] = Start-Dashboard -Name "Zeus-Dash" -Script "backtesting\zeus_dashboard.py" -Port 8502 -Log (Join-Path $LogDir "zeus_dash.log")
    Start-Sleep -Seconds 3
} else {
    Write-Host "[Zeus-Dash] Port 8502 already in use - skipping." -ForegroundColor Yellow
}

if (Test-PortFree 8503) {
    $pids["hermes_dash"] = Start-Dashboard -Name "Hermes-Dash" -Script "ui\hermes_dashboard.py" -Port 8503 -Log (Join-Path $LogDir "hermes_dash.log")
    Start-Sleep -Seconds 3
} else {
    Write-Host "[Hermes-Dash] Port 8503 already in use - skipping." -ForegroundColor Yellow
}

if (-not $SkipHermesBot) {
    $hermesParams = @(
        "live_bot\run_hermes.py"
        "--tf"
        "M5"
        "--lot"
        "0.01"
    )
    $pids["hermes_bot"] = Start-Proc -Name "Hermes-Bot" -Exe $PYTHON -ProcParams $hermesParams -Cwd $ROOT -Log (Join-Path $LogDir "hermes_bot.log")
    Start-Sleep -Seconds 2
}

if (Test-PortFree 8504) {
    $pids["hermes_academy_dash"] = Start-Dashboard -Name "Hermes-Academy-Dash" -Script "ui\hermes_academy_dashboard.py" -Port 8504 -Log (Join-Path $LogDir "hermes_academy_dash.log")
    Start-Sleep -Seconds 3
} else {
    Write-Host "[Hermes-Academy-Dash] Port 8504 already in use - skipping." -ForegroundColor Yellow
}

if (Test-PortFree 8505) {
    $pids["hermes_return_dash"] = Start-Dashboard -Name "Hermes-Return-Dash" -Script "ui\hermes_return_dashboard.py" -Port 8505 -Log (Join-Path $LogDir "hermes_return_dash.log")
    Start-Sleep -Seconds 3
} else {
    Write-Host "[Hermes-Return-Dash] Port 8505 already in use - skipping." -ForegroundColor Yellow
}

if (Test-PortFree 8506) {
    $pids["hermes_pattern_context_dash"] = Start-Dashboard -Name "Hermes-Pattern-Context-Dash" -Script "ui\hermes_pattern_context_dashboard.py" -Port 8506 -Log (Join-Path $LogDir "hermes_pattern_context_dash.log")
    Start-Sleep -Seconds 3
} else {
    Write-Host "[Hermes-Pattern-Context-Dash] Port 8506 already in use - skipping." -ForegroundColor Yellow
}

if (Test-PortFree 8507) {
    $pids["olympus_observability_dash"] = Start-Dashboard -Name "Olympus-Observability-Dash" -Script "ui\olympus_observability_dashboard.py" -Port 8507 -Log (Join-Path $LogDir "olympus_observability_dash.log")
    Start-Sleep -Seconds 3
} else {
    Write-Host "[Olympus-Observability-Dash] Port 8507 already in use - skipping." -ForegroundColor Yellow
}

if (Test-PortFree 8508) {
    $pids["prometheus_execution_academy_dash"] = Start-Dashboard -Name "Prometheus-Execution-Academy-Dash" -Script "ui\prometheus_execution_academy_dashboard.py" -Port 8508 -Log (Join-Path $LogDir "prometheus_execution_academy_dash.log")
    Start-Sleep -Seconds 3
} else {
    Write-Host "[Prometheus-Execution-Academy-Dash] Port 8508 already in use - skipping." -ForegroundColor Yellow
}

if (Test-PortFree 8509) {
    $pids["prometheus_academy_observability_dash"] = Start-Dashboard -Name "Prometheus-Academy-Observability-Dash" -Script "ui\prometheus_academy_observability_dashboard.py" -Port 8509 -Log (Join-Path $LogDir "prometheus_academy_observability_dash.log")
    Start-Sleep -Seconds 3
} else {
    Write-Host "[Prometheus-Academy-Observability-Dash] Port 8509 already in use - skipping." -ForegroundColor Yellow
}

if (Test-PortFree 8510) {
    $pids["prometheus_evolution_dash"] = Start-Dashboard -Name "Prometheus-Evolution-Dash" -Script "ui\prometheus_evolution_dashboard.py" -Port 8510 -Log (Join-Path $LogDir "prometheus_evolution_dash.log")
    Start-Sleep -Seconds 3
} else {
    Write-Host "[Prometheus-Evolution-Dash] Port 8510 already in use - skipping." -ForegroundColor Yellow
}

if (Test-PortFree 8511) {
    $pids["knowledge_growth_dash"] = Start-Dashboard -Name "Olympus-Command-Center" -Script "ui\knowledge_growth_dashboard.py" -Port 8511 -Log (Join-Path $LogDir "knowledge_growth_dash.log")
    Start-Sleep -Seconds 3
} else {
    Write-Host "[Olympus-Command-Center] Port 8511 already in use - skipping." -ForegroundColor Yellow
}

if (-not $SkipTunnels) {
    if (-not (Test-Path $CF)) {
        Write-Host "[CF] cloudflared not found at $CF - skipping tunnels." -ForegroundColor Red
    } else {
        foreach ($port in @(8501, 8502, 8503, 8504, 8505, 8506, 8507, 8508, 8509, 8510, 8511)) {
            $pids["cf_$port"] = Start-Tunnel -Port $port -Log (Join-Path $LogDir "cf_$port.log")
            Start-Sleep -Seconds 2
        }

        Write-Host "`nWaiting 20s for Cloudflare tunnel negotiation..." -ForegroundColor Cyan
        Start-Sleep -Seconds 20

        Write-Host "`n=== CLOUDFLARE PUBLIC URLS ===" -ForegroundColor Green
        foreach ($port in @(8501, 8502, 8503, 8504, 8505, 8506, 8507, 8508, 8509, 8510, 8511)) {
            $log = Join-Path $LogDir "cf_$port.log"
            $url = Get-Content $log -ErrorAction SilentlyContinue | Select-String "trycloudflare.com" | Select-Object -Last 1
            $name = @{
                8501 = "Prometheus"
                8502 = "Zeus"
                8503 = "Hermes"
                8504 = "Hermes Academy"
                8505 = "Hermes Return Intelligence"
                8506 = "Hermes Pattern Context"
                8507 = "Olympus Observability"
                8508 = "Prometheus Execution Academy"
                8509 = "Prometheus Academy Observability"
                8510 = "Prometheus Evolution"
                8511 = "Olympus Command Center"
            }[$port]
            if ($url) {
                Write-Host "$name ($port): $url" -ForegroundColor Yellow
                $pids["cf_url_$port"] = "$url".Trim()
            } else {
                Write-Host "$name ($port): URL not yet available (check logs\cf_$port.log)" -ForegroundColor Red
            }
        }
    }
}

$pids | ConvertTo-Json | Set-Content -Path $statusFile -Encoding UTF8
Write-Host "`nAll services started. PID map saved to: $statusFile" -ForegroundColor Green
