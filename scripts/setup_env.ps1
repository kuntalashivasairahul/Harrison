param(
    [string]$PythonCommand = "py",
    [string]$PythonVersionArgument = "-3.12"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

try {
    & $PythonCommand $PythonVersionArgument --version 2>$null
    if ($LASTEXITCODE -ne 0) {
        throw "Python launcher returned exit code $LASTEXITCODE."
    }
} catch {
    throw "Python 3.12 was not found. Install Python 3.12, then run this script again."
}

if (-not (Test-Path ".venv312")) {
    & $PythonCommand $PythonVersionArgument -m venv .venv312
}

& .venv312\Scripts\python.exe -m pip install --upgrade pip
& .venv312\Scripts\python.exe -m pip install -r backend\requirements.txt
& .venv312\Scripts\python.exe --version
