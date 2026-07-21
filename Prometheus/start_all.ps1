# =============================================================================
# Prometheus - Full Stack Startup Script
# Starts: Prometheus bot, Hermes bot, three consolidated command centres, and
#         optional tunnels. Legacy dashboards remain available as lazy pages.
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
$publicLinks = @()

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
    $pids["prometheus_command_center"] = Start-Dashboard -Name "Prometheus-Command-Center" -Script "ui\prometheus_command_center.py" -Port 8501 -Log (Join-Path $LogDir "prometheus_command_center.log")
    Start-Sleep -Seconds 3
} else {
    Write-Host "[Prometheus-Command-Center] Port 8501 already in use - skipping." -ForegroundColor Yellow
}

if (Test-PortFree 8503) {
    $pids["hermes_command_center"] = Start-Dashboard -Name "Hermes-Command-Center" -Script "ui\hermes_command_center.py" -Port 8503 -Log (Join-Path $LogDir "hermes_command_center.log")
    Start-Sleep -Seconds 3
} else {
    Write-Host "[Hermes-Command-Center] Port 8503 already in use - skipping." -ForegroundColor Yellow
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

if (Test-PortFree 8511) {
    $pids["olympus_command_center"] = Start-Dashboard -Name "Olympus-Command-Center" -Script "ui\olympus_command_center.py" -Port 8511 -Log (Join-Path $LogDir "olympus_command_center.log")
    Start-Sleep -Seconds 3
} else {
    Write-Host "[Olympus-Command-Center] Port 8511 already in use - skipping." -ForegroundColor Yellow
}

if (-not $SkipTunnels) {
    if (-not (Test-Path $CF)) {
        Write-Host "[CF] cloudflared not found at $CF - skipping tunnels." -ForegroundColor Red
    } else {
        foreach ($port in @(8501, 8503, 8511)) {
            $pids["cf_$port"] = Start-Tunnel -Port $port -Log (Join-Path $LogDir "cf_$port.log")
            Start-Sleep -Seconds 2
        }

        Write-Host "`nWaiting 20s for Cloudflare tunnel negotiation..." -ForegroundColor Cyan
        Start-Sleep -Seconds 20

        Write-Host "`n=== CLOUDFLARE PUBLIC URLS ===" -ForegroundColor Green
        foreach ($port in @(8501, 8503, 8511)) {
            $log = Join-Path $LogDir "cf_$port.log"
            $url = Get-Content $log -ErrorAction SilentlyContinue | Select-String "trycloudflare.com" | Select-Object -Last 1
            $name = @{
                8501 = "Prometheus Trading Command Center"
                8503 = "Hermes Execution and Learning Center"
                8511 = "Olympus Governance and Research Center"
            }[$port]
            if ($url) {
                $urlMatch = [regex]::Match("$url", 'https://[A-Za-z0-9-]+\.trycloudflare\.com')
                if (-not $urlMatch.Success) {
                    Write-Host "$name ($port): malformed URL in tunnel log" -ForegroundColor Red
                    continue
                }
                $cleanUrl = $urlMatch.Value
                Write-Host "$name ($port): $cleanUrl" -ForegroundColor Yellow
                $pids["cf_url_$port"] = $cleanUrl
                $publicLinks += "$name=$cleanUrl"
            } else {
                Write-Host "$name ($port): URL not yet available (check logs\cf_$port.log)" -ForegroundColor Red
            }
        }

        if ($publicLinks.Count -gt 0) {
            if ($env:TELEGRAM_BOT_TOKEN -and $env:TELEGRAM_CHAT_ID) {
                $telegramArgs = @("scripts\notify_telegram_links.py")
                foreach ($link in $publicLinks) {
                    $telegramArgs += @("--link", $link)
                }
                & $PYTHON @telegramArgs
                if ($LASTEXITCODE -ne 0) {
                    Write-Host "[Telegram] Dashboard link notification failed; see the message above." -ForegroundColor Red
                }
            } else {
                Write-Host "[Telegram] Links not sent: set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID." -ForegroundColor Yellow
            }
        }
    }
}

$pids | ConvertTo-Json | Set-Content -Path $statusFile -Encoding UTF8
Write-Host "`nAll services started. PID map saved to: $statusFile" -ForegroundColor Green
