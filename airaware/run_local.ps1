$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

$projectRoot = $PSScriptRoot
$backendPath = Join-Path $projectRoot "backend"
$frontendPath = Join-Path $projectRoot "frontend"
$logPath = Join-Path $projectRoot "logs"
$venvPython = Join-Path $backendPath ".venv\Scripts\python.exe"
$pythonExe = if (Test-Path -LiteralPath $venvPython) { $venvPython } else { "python" }

$backendProcess = $null
$frontendProcess = $null
$startedBackend = $false
$startedFrontend = $false

function Test-AirAwareUrl {
    param([string]$Url)
    try {
        $response = Invoke-WebRequest -UseBasicParsing -Uri $Url -TimeoutSec 2
        return $response.StatusCode -eq 200
    } catch {
        return $false
    }
}

function Wait-AirAwareUrl {
    param([string]$Url, [int]$Seconds = 30)
    for ($attempt = 0; $attempt -lt $Seconds; $attempt++) {
        if (Test-AirAwareUrl -Url $Url) { return $true }
        Start-Sleep -Seconds 1
    }
    return $false
}

function Show-AirAwareLog {
    param([string]$File)
    if (Test-Path -LiteralPath $File) {
        Write-Host "`n--- $File ---" -ForegroundColor DarkYellow
        Get-Content -LiteralPath $File -Tail 30
    }
}

New-Item -ItemType Directory -Force -Path $logPath | Out-Null
$backendOut = Join-Path $logPath "backend.out.log"
$backendErr = Join-Path $logPath "backend.error.log"
$frontendOut = Join-Path $logPath "frontend.out.log"
$frontendErr = Join-Path $logPath "frontend.error.log"

try {
    Write-Host "Starting AirAware locally..." -ForegroundColor Cyan

    if (Test-AirAwareUrl -Url "http://127.0.0.1:8000/health") {
        Write-Host "[OK] Backend is already running." -ForegroundColor Green
    } else {
        $backendProcess = Start-Process -FilePath $pythonExe `
            -ArgumentList "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", "8000" `
            -WorkingDirectory $backendPath -WindowStyle Hidden -PassThru `
            -RedirectStandardOutput $backendOut -RedirectStandardError $backendErr
        $startedBackend = $true
        if (-not (Wait-AirAwareUrl -Url "http://127.0.0.1:8000/health")) {
            throw "Backend did not become ready."
        }
        Write-Host "[OK] Backend started." -ForegroundColor Green
    }

    if (Test-AirAwareUrl -Url "http://127.0.0.1:5173/") {
        Write-Host "[OK] Frontend is already running." -ForegroundColor Green
    } else {
        $frontendProcess = Start-Process -FilePath "npm.cmd" `
            -ArgumentList "run", "dev", "--", "--host", "127.0.0.1" `
            -WorkingDirectory $frontendPath -WindowStyle Hidden -PassThru `
            -RedirectStandardOutput $frontendOut -RedirectStandardError $frontendErr
        $startedFrontend = $true
        if (-not (Wait-AirAwareUrl -Url "http://127.0.0.1:5173/")) {
            throw "Frontend did not become ready."
        }
        Write-Host "[OK] Frontend started." -ForegroundColor Green
    }

    Write-Host ""
    Write-Host "Backend: http://127.0.0.1:8000"
    Write-Host "Frontend: http://localhost:5173"
    Write-Host "Press Ctrl+C to stop processes started by this script." -ForegroundColor DarkGray

    while ($true) {
        Start-Sleep -Seconds 2
        if (-not (Test-AirAwareUrl -Url "http://127.0.0.1:8000/health")) {
            throw "Backend stopped responding."
        }
        if (-not (Test-AirAwareUrl -Url "http://127.0.0.1:5173/")) {
            throw "Frontend stopped responding."
        }
    }
} catch {
    Write-Host "`nAirAware could not stay running: $($_.Exception.Message)" -ForegroundColor Red
    Show-AirAwareLog -File $backendErr
    Show-AirAwareLog -File $frontendErr
    Read-Host "Press Enter to close"
} finally {
    if ($startedBackend -and $null -ne $backendProcess) {
        $backendProcess.Refresh()
        if (-not $backendProcess.HasExited) { Stop-Process -Id $backendProcess.Id }
    }
    if ($startedFrontend -and $null -ne $frontendProcess) {
        $frontendProcess.Refresh()
        if (-not $frontendProcess.HasExited) { Stop-Process -Id $frontendProcess.Id }
    }
}
