# =============================================================================
# Prometheus — Register Windows Scheduled Tasks
# Registers:
#   "PrometheusStartAll"  — every Monday at 01:00 (UTC+2 local, adjust if needed)
#   "PrometheusStopAll"   — every Friday at 23:00
# Run ONCE as Administrator:
#   powershell -ExecutionPolicy Bypass -File register_tasks.ps1
# =============================================================================

$ErrorActionPreference = "Stop"
$ROOT = Split-Path -Parent $MyInvocation.MyCommand.Definition

$startScript = Join-Path $ROOT "start_all.ps1"
$stopScript  = Join-Path $ROOT "stop_all.ps1"
$ps          = "C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe"

if (-not (Test-Path $startScript)) { throw "start_all.ps1 not found at $startScript" }
if (-not (Test-Path $stopScript))  { throw "stop_all.ps1 not found at $stopScript"   }

# ── Helper ───────────────────────────────────────────────────────────────────
function Register-PrometheusTask {
    param($TaskName, $Script, $DayOfWeek, $Hour, $Minute)

    $action = New-ScheduledTaskAction `
        -Execute $ps `
        -Argument "-NonInteractive -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$Script`"" `
        -WorkingDirectory $ROOT

    $trigger = New-ScheduledTaskTrigger `
        -Weekly `
        -DaysOfWeek $DayOfWeek `
        -At ([datetime]"$(Get-Date -Format yyyy-MM-dd) $Hour`:$Minute`:00")

    # Run interactively as current user (no admin needed)
    $principal = New-ScheduledTaskPrincipal `
        -UserId $env:USERNAME `
        -LogonType Interactive

    $settings = New-ScheduledTaskSettingsSet `
        -ExecutionTimeLimit (New-TimeSpan -Hours 2) `
        -RestartCount 3 `
        -RestartInterval (New-TimeSpan -Minutes 5) `
        -StartWhenAvailable `
        -MultipleInstances IgnoreNew

    # Remove existing if present
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue

    Register-ScheduledTask `
        -TaskName $TaskName `
        -Action $action `
        -Trigger $trigger `
        -Principal $principal `
        -Settings $settings `
        -Description "Prometheus trading system auto-start/stop" | Out-Null

    Write-Host "Registered task: $TaskName  ($DayOfWeek @ $Hour`:$Minute)" -ForegroundColor Green
}

# ── Register ──────────────────────────────────────────────────────────────────
Write-Host "Registering scheduled tasks..." -ForegroundColor Cyan

Register-PrometheusTask -TaskName "PrometheusStartAll" `
    -Script $startScript -DayOfWeek "Monday" -Hour 1 -Minute 0

Register-PrometheusTask -TaskName "PrometheusStopAll" `
    -Script $stopScript -DayOfWeek "Friday" -Hour 23 -Minute 0

Write-Host "`nTasks registered successfully." -ForegroundColor Green
Write-Host "To verify: Get-ScheduledTask -TaskName 'PrometheusStartAll','PrometheusStopAll' | Format-List"
Write-Host "To run now: Start-ScheduledTask -TaskName 'PrometheusStartAll'"
