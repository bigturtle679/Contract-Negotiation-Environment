$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
$env:PYTHONPATH = $Root
$env:USE_DIRECT_ENV = "1"
if (-not $env:HF_TOKEN) { $env:HF_TOKEN = "" }
python inference.py
