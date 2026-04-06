$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot\..

docker build -t contract-negotiation-env .
$cid = docker run -d -p 7860:7860 contract-negotiation-env
try {
    Start-Sleep -Seconds 3
    $r = Invoke-WebRequest -Uri "http://localhost:7860/reset" -Method POST -UseBasicParsing
    if ($r.StatusCode -ne 200) { throw "Expected 200 from /reset, got $($r.StatusCode)" }
    if (Get-Command openenv -ErrorAction SilentlyContinue) {
        openenv validate .
        openenv validate --url http://localhost:7860
    } else {
        Write-Host "openenv CLI not on PATH; skipped openenv validate"
    }
} finally {
    docker stop $cid | Out-Null
    docker rm $cid | Out-Null
}
