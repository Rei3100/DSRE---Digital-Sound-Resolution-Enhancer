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

Write-Host "[build] fetch ffmpeg (BtbN win64 gpl, latest)"
$ffStage = Join-Path $PSScriptRoot "_ffmpeg"
if (-not (Test-Path (Join-Path $ffStage "ffmpeg.exe"))) {
    $ffUrl = "https://github.com/BtbN/FFmpeg-Builds/releases/latest/download/ffmpeg-master-latest-win64-gpl.zip"
    $ffZip = Join-Path $PSScriptRoot "ffmpeg.zip"
    $ffExtract = Join-Path $PSScriptRoot "ffmpeg_extract"
    if (Test-Path $ffExtract) { Remove-Item $ffExtract -Recurse -Force }
    Invoke-WebRequest -Uri $ffUrl -OutFile $ffZip
    Expand-Archive -Path $ffZip -DestinationPath $ffExtract -Force
    $ff = Get-ChildItem $ffExtract -Recurse -Filter ffmpeg.exe | Select-Object -First 1
    if (-not $ff) { throw "ffmpeg.exe not found in BtbN archive" }
    New-Item -ItemType Directory -Path $ffStage -Force | Out-Null
    Copy-Item $ff.FullName (Join-Path $ffStage "ffmpeg.exe") -Force
    Remove-Item $ffZip -Force -ErrorAction SilentlyContinue
    Remove-Item $ffExtract -Recurse -Force -ErrorAction SilentlyContinue
}

Write-Host "[build] pyinstaller DSRE.spec"
& $pyi DSRE.spec --clean --noconfirm
if ($LASTEXITCODE -ne 0) { throw "pyinstaller failed with exit $LASTEXITCODE" }

$dist = Join-Path $PSScriptRoot "dist\DSRE"

Write-Host "[build] stage ffmpeg into _internal/ffmpeg"
$ffDst = Join-Path $dist "_internal\ffmpeg"
New-Item -ItemType Directory -Path $ffDst -Force | Out-Null
Copy-Item (Join-Path $ffStage "ffmpeg.exe") (Join-Path $ffDst "ffmpeg.exe") -Force

$required = @("_internal\numpy", "_internal\scipy", "_internal\librosa", "_internal\numba", "_internal\llvmlite", "_internal\resampy", "_internal\soundfile", "_internal\ffmpeg\ffmpeg.exe", "DSRE.exe")
foreach ($r in $required) {
    $p = Join-Path $dist $r
    if (-not (Test-Path $p)) { throw "MISSING after build: $p" }
}

Write-Host "[build] structural smoke OK"

Write-Host "[build] runtime selftest (--selftest)"
$exe = Join-Path $dist "DSRE.exe"
# windowed exe は & 演算子だと detach するので Start-Process -Wait -PassThru で完了待機
$proc = Start-Process -FilePath $exe -ArgumentList "--selftest" -Wait -PassThru
$code = $proc.ExitCode
$log = Join-Path $dist "selftest.log"
if (Test-Path $log) {
    Write-Host "---- selftest.log ----"
    Get-Content $log
    Write-Host "----------------------"
}
if ($code -ne 0) { throw "runtime selftest failed with exit $code" }

Write-Host "[build] OK -> $exe"
Write-Host "[build] deploy with: powershell -File $PSScriptRoot\deploy.ps1 -Source '$dist'"
