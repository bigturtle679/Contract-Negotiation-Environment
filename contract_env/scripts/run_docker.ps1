$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

docker build -t contract-negotiation-env:latest .
docker rm -f contract-negotiation-run 2>$null
docker run -d --name contract-negotiation-run -p 7860:7860 contract-negotiation-env:latest
Start-Sleep -Seconds 2

$env:PYTHONPATH = $Root
$env:USE_DIRECT_ENV = ""
$env:BASE_URL = "http://127.0.0.1:7860"
python inference.py

docker rm -f contract-negotiation-run
