#Requires -Version 5.1
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$venv = Join-Path $PSScriptRoot ".venv"
if (-not (Test-Path $venv)) {
    Write-Host "[build] create venv (py -3.11 -m venv .venv)"
    py -3.11 -m venv .venv
}

$py = Join-Path $venv "Scripts\python.exe"
$pip = Join-Path $venv "Scripts\pip.exe"
$pyi = Join-Path $venv "Scripts\pyinstaller.exe"

Write-Host "[build] upgrade pip + install deps"
& $py -m pip install --upgrade pip
& $pip install -r requirements.txt
& $pip install -r requirements-dev.txt

Write-Host "[build] clean old dist/build"
foreach ($d in @("build", "dist", "release")) {
    $p = Join-Path $PSScriptRoot $d
    if (Test-Path $p) { Remove-Item $p -Recurse -Force }
}

Write-Host "[build] pyinstaller DSRE.spec"
& $pyi DSRE.spec --clean --noconfirm
if ($LASTEXITCODE -ne 0) { throw "pyinstaller failed with exit $LASTEXITCODE" }

$dist = Join-Path $PSScriptRoot "dist\DSRE"
$required = @("_internal\numpy", "_internal\scipy", "_internal\librosa", "_internal\numba", "_internal\llvmlite", "_internal\resampy", "_internal\soundfile", "DSRE.exe")
foreach ($r in $required) {
    $p = Join-Path $dist $r
    if (-not (Test-Path $p)) { throw "MISSING after build: $p" }
}

Write-Host "[build] OK -> $dist\DSRE.exe"
Write-Host "[build] deploy with: powershell -File $PSScriptRoot\deploy.ps1 -Source '$dist'"
