#Requires -Version 5.1
<#
.SYNOPSIS
    Start the Prometheus Streamlit dashboard and expose it via Cloudflare Tunnel.

.DESCRIPTION
    Run this script any time you want remote access from your phone or any device.
    A fresh public URL is generated on each run (no account required).

    The tunnel stays alive as a background Windows process until you stop it:
        Get-Process cloudflared | Stop-Process -Force

.NOTES
    Requirements:
      - cloudflared.exe at %LOCALAPPDATA%\cloudflared\cloudflared.exe
        (download: https://github.com/cloudflare/cloudflared/releases)
      - Python venv : C:\Users\Chaba\Documents\tradingBots\.venv
      - Prometheus  : C:\Users\Chaba\Documents\tradingBots\Prometheus
#>

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# ─── Configuration ────────────────────────────────────────────────────────────
$STREAMLIT   = 'C:\Users\Chaba\Documents\tradingBots\.venv\Scripts\streamlit.exe'
$PROMETHEUS  = 'C:\Users\Chaba\Documents\tradingBots\Prometheus'
$CLOUDFLARED = "$env:LOCALAPPDATA\cloudflared\cloudflared.exe"
$PORT        = 8501
$LOG_OUT     = Join-Path $env:TEMP 'cf-prometheus-tunnel.log'
$LOG_ERR     = Join-Path $env:TEMP 'cf-prometheus-tunnel.err'

# ─── Helper: check if a TCP port is in LISTEN state ──────────────────────────
function Test-PortListening ([int]$port) {
    $null -ne (
        Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue |
        Select-Object -First 1
    )
}

Write-Host ''
Write-Host '  Prometheus Remote Access Setup' -ForegroundColor Cyan
Write-Host '  ------------------------------' -ForegroundColor DarkGray
Write-Host ''

# Optional dashboard selector (backward compatible default remains 8501)
if ($env:PROMETHEUS_TUNNEL_PORT) {
    try { $PORT = [int]$env:PROMETHEUS_TUNNEL_PORT } catch {}
}
if ($PORT -eq 8511) {
    Write-Host '  Target Dashboard: Olympus Command Center (8511)' -ForegroundColor Green
    $DASHBOARD_SCRIPT = 'ui\olympus_command_center.py'
} elseif ($PORT -eq 8503) {
    Write-Host '  Target Dashboard: Hermes Command Center (8503)' -ForegroundColor Green
    $DASHBOARD_SCRIPT = 'ui\hermes_command_center.py'
} else {
    Write-Host "  Target Dashboard: Prometheus (port $PORT)" -ForegroundColor Green
    $DASHBOARD_SCRIPT = 'ui\prometheus_command_center.py'
}

# ─── Step 1: Validate cloudflared ────────────────────────────────────────────
if (-not (Test-Path $CLOUDFLARED)) {
    Write-Host '[ERROR] cloudflared not found.' -ForegroundColor Red
    Write-Host "        Expected: $CLOUDFLARED" -ForegroundColor Red
    Write-Host '        Download: https://github.com/cloudflare/cloudflared/releases' -ForegroundColor Red
    exit 1
}

# ─── Step 2: Start Streamlit if not already running ──────────────────────────
if (-not (Test-PortListening $PORT)) {
    Write-Host "[1/4] Starting Streamlit on port $PORT..." -ForegroundColor Cyan

    Start-Process -FilePath $STREAMLIT `
        -ArgumentList @(
            'run', $DASHBOARD_SCRIPT,
            '--server.port',                  $PORT,
            '--server.address',               '127.0.0.1',
            '--server.headless',              'true',
            '--server.enableCORS',            'false',
            '--server.enableXsrfProtection',  'false'
        ) `
        -WorkingDirectory $PROMETHEUS `
        -WindowStyle Minimized

    Write-Host '         Waiting for Streamlit.' -NoNewline -ForegroundColor DarkGray
    $waited = 0
    while (-not (Test-PortListening $PORT) -and $waited -lt 30) {
        Start-Sleep -Seconds 2; $waited += 2; Write-Host '.' -NoNewline
    }
    Write-Host ''

    if (-not (Test-PortListening $PORT)) {
        Write-Host '[ERROR] Streamlit did not start within 30 s.' -ForegroundColor Red
        exit 1
    }
    Write-Host "[1/4] Streamlit is ready on port $PORT." -ForegroundColor Green
} else {
    Write-Host "[1/4] Streamlit already running on port $PORT." -ForegroundColor Green
}

# ─── Step 3: Clear stale cloudflared processes ────────────────────────────────
Write-Host '[2/3] Clearing any existing tunnel processes...' -ForegroundColor Cyan
Get-Process cloudflared -ErrorAction SilentlyContinue |
    Stop-Process -Force -ErrorAction SilentlyContinue
Start-Sleep -Milliseconds 700

# ─── Step 4: Start cloudflared quick tunnel (no account / API needed) ─────────
Write-Host '[3/3] Starting Cloudflare quick tunnel...' -ForegroundColor Cyan

'' | Set-Content -Path $LOG_OUT
'' | Set-Content -Path $LOG_ERR

$cfProc = Start-Process -FilePath $CLOUDFLARED `
    -ArgumentList @(
        'tunnel',
        '--no-autoupdate',
        '--url', "http://127.0.0.1:$PORT"
    ) `
    -WorkingDirectory $env:TEMP `
    -RedirectStandardOutput $LOG_OUT `
    -RedirectStandardError  $LOG_ERR `
    -WindowStyle Hidden `
    -PassThru

Write-Host '         Waiting for public URL.' -NoNewline -ForegroundColor DarkGray
$deadline        = (Get-Date).AddSeconds(45)
$connected       = $false
$publicHostname  = $null

while ((Get-Date) -lt $deadline) {
    Start-Sleep -Seconds 2
    Write-Host '.' -NoNewline

    $logContent = ''
    if (Test-Path $LOG_OUT) { $logContent += Get-Content $LOG_OUT -Raw -ErrorAction SilentlyContinue }
    if (Test-Path $LOG_ERR) { $logContent += Get-Content $LOG_ERR -Raw -ErrorAction SilentlyContinue }

    # Quick tunnel prints the URL in a banner line
    $urlMatch = [regex]::Match($logContent, 'https://[a-z0-9\-]+\.trycloudflare\.com')
    if ($urlMatch.Success) {
        $publicHostname = $urlMatch.Value -replace '^https://', ''
        $connected = $true
        break
    }

    if ($cfProc.HasExited) {
        Write-Host ''
        Write-Host '[ERROR] cloudflared exited unexpectedly.' -ForegroundColor Red
        Write-Host "        Check logs: $LOG_ERR" -ForegroundColor Red
        Get-Content $LOG_ERR -ErrorAction SilentlyContinue | Select-Object -Last 20 | Write-Host
        exit 1
    }
}
Write-Host ''

# ─── Step 6: Print result ─────────────────────────────────────────────────────
$url  = "https://$publicHostname"
$pad  = 62

Write-Host ''
if ($connected) {
    Write-Host '  +--------------------------------------------------------------+' -ForegroundColor Green
    Write-Host '  |   PROMETHEUS DASHBOARD -- ONLINE                             |' -ForegroundColor Green
    Write-Host '  +--------------------------------------------------------------+' -ForegroundColor Green
    Write-Host ("  |   " + $url.PadRight($pad) + "|") -ForegroundColor Yellow
    Write-Host '  |                                                              |' -ForegroundColor Green
    Write-Host '  |   Open this link on your phone -- no account needed.         |' -ForegroundColor Green
    Write-Host '  |   Tunnel runs until you stop cloudflared (see below).        |' -ForegroundColor Green
    Write-Host '  +--------------------------------------------------------------+' -ForegroundColor Green
} else {
    Write-Host '  [WARNING] Edge connection not confirmed within timeout.' -ForegroundColor Yellow
    Write-Host "            The tunnel may still be starting. Try: $url" -ForegroundColor Yellow
}

Write-Host ''
Write-Host "  Tunnel logs : $LOG_ERR" -ForegroundColor DarkGray
Write-Host '  Stop tunnel : Get-Process cloudflared | Stop-Process -Force' -ForegroundColor DarkGray
Write-Host ''
