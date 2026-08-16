param(
    [string]$HostAddress = "127.0.0.1",
    [int]$Port = 8000
)

$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$backendDir = Join-Path $projectRoot "backend"
$venvPython = Join-Path (Join-Path $projectRoot "..\dataset_CUDA\.venv\Scripts") "python.exe"

if (-not (Test-Path $venvPython)) {
    Write-Error "Python not found: $venvPython"
    Write-Host "Create the venv first in dataset_CUDA/.venv or update this script path." -ForegroundColor Yellow
    exit 1
}

# Runtime tuning defaults for the local model.
$env:USE_4BIT = if ($env:USE_4BIT) { $env:USE_4BIT } else { "false" }
$env:MAX_NEW_TOKENS = if ($env:MAX_NEW_TOKENS) { $env:MAX_NEW_TOKENS } else { "120" }
$env:TEMPERATURE = if ($env:TEMPERATURE) { $env:TEMPERATURE } else { "0.6" }
$env:TOP_P = if ($env:TOP_P) { $env:TOP_P } else { "0.9" }

# Ensure VibeVoice subprocess uses the same Python environment.
$env:VIBEVOICE_PYTHON = $venvPython

Write-Host "Starting LLM backend..." -ForegroundColor Cyan
Write-Host "Backend directory: $backendDir"
Write-Host "Python: $venvPython"
Write-Host "Host: $HostAddress  Port: $Port"

Set-Location $backendDir
& $venvPython -m uvicorn main:app --host $HostAddress --port $Port
