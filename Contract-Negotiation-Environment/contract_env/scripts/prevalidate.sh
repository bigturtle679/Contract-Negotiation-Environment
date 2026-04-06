#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "Building Docker image..."
docker build -t contract-negotiation-env .

docker rm -f contract-negotiation-env-test 2>/dev/null || true

echo "Starting container on port 7860..."
docker run -d --name contract-negotiation-env-test -p 7860:7860 contract-negotiation-env

sleep 3

echo "POST /reset ..."
code="$(curl -s -o /dev/null -w '%{http_code}' -X POST http://127.0.0.1:7860/reset -H 'Content-Type: application/json' -d '{}')"
if [[ "$code" != "200" ]]; then
  docker logs contract-negotiation-env-test || true
  docker rm -f contract-negotiation-env-test
  echo "Expected HTTP 200 from /reset, got $code" >&2
  exit 1
fi

echo "OK: /reset returned 200"

if command -v openenv >/dev/null 2>&1; then
  echo "openenv validate..."
  openenv validate --verbose "$ROOT" || true
  openenv validate --url "http://127.0.0.1:7860" || true
fi

docker rm -f contract-negotiation-env-test

echo "Pre-validation finished."
