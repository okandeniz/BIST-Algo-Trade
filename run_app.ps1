param(
    [switch]$Headless,
    [ValidateRange(1, 65535)]
    [int]$BackendPort = 8000,
    [ValidateRange(1, 65535)]
    [int]$FrontendPort = 8501
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $ProjectRoot

$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$Python = if (Test-Path -LiteralPath $VenvPython) {
    $VenvPython
} else {
    (Get-Command python -ErrorAction Stop).Source
}

$ApiUrl = "http://127.0.0.1:$BackendPort"
$HealthUrl = "$ApiUrl/health"
$BackendProcess = $null
$StartedBackend = $false

function Test-RobotApi {
    try {
        $Response = Invoke-RestMethod `
            -Uri $HealthUrl `
            -TimeoutSec 2
        return $Response.status -eq "ok"
    } catch {
        return $false
    }
}

try {
    if (-not (Test-RobotApi)) {
        $LogDirectory = Join-Path $ProjectRoot "results\app"
        New-Item `
            -ItemType Directory `
            -Force `
            -Path $LogDirectory | Out-Null

        $BackendOutput = Join-Path $LogDirectory "backend.stdout.log"
        $BackendError = Join-Path $LogDirectory "backend.stderr.log"

        Write-Host "BIST100 Robot backend başlatılıyor..."
        $BackendProcess = Start-Process `
            -FilePath $Python `
            -ArgumentList @(
                "-m",
                "uvicorn",
                "backend.app.main:app",
                "--host",
                "127.0.0.1",
                "--port",
                "$BackendPort"
            ) `
            -WorkingDirectory $ProjectRoot `
            -WindowStyle Hidden `
            -RedirectStandardOutput $BackendOutput `
            -RedirectStandardError $BackendError `
            -PassThru
        $StartedBackend = $true

        $Ready = $false
        for ($Attempt = 0; $Attempt -lt 40; $Attempt++) {
            if ($BackendProcess.HasExited) {
                throw (
                    "Backend başlatılamadı. Logları kontrol edin: " +
                    "$BackendError"
                )
            }
            if (Test-RobotApi) {
                $Ready = $true
                break
            }
            Start-Sleep -Milliseconds 500
        }

        if (-not $Ready) {
            throw "Backend 20 saniye içinde hazır olmadı."
        }
    } else {
        Write-Host "Çalışan BIST100 Robot backend kullanılacak."
    }

    $env:API_URL = $ApiUrl
    $HeadlessValue = if ($Headless) { "true" } else { "false" }

    Write-Host "Arayüz açılıyor: http://127.0.0.1:$FrontendPort"
    & $Python `
        -m streamlit run frontend/app.py `
        --server.address 127.0.0.1 `
        --server.port $FrontendPort `
        --server.headless $HeadlessValue
} finally {
    if (
        $StartedBackend `
        -and $null -ne $BackendProcess `
        -and -not $BackendProcess.HasExited
    ) {
        Write-Host "Backend kapatılıyor..."
        Stop-Process -Id $BackendProcess.Id -Force
    }
}
