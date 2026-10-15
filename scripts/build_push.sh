#!/usr/bin/env bash
# Build and push the 4 images. REGISTRY examples:
#   k3d-demandops-registry.localhost:5111/demandops   (local, default in `make images`)
#   harbor.example.ir/demandops                        (your Harbor)
#   ghcr.io/<user>                                     (GitHub Container Registry)
set -euo pipefail
: "${REGISTRY:?set REGISTRY}"
TAG="${TAG:-$(git rev-parse --short HEAD 2>/dev/null || echo latest)}"
PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.org/simple}"

build() { # name dockerfile [extra args]
  local name=$1 file=$2; shift 2
  docker build -f "infra/docker/${file}" --build-arg "PIP_INDEX_URL=${PIP_INDEX_URL}" "$@" \
    -t "${REGISTRY}/${name}:${TAG}" -t "${REGISTRY}/${name}:latest" .
  docker push "${REGISTRY}/${name}:${TAG}"
  docker push "${REGISTRY}/${name}:latest"
}

build demandops-serving   Dockerfile --target serving
build demandops-jobs      Dockerfile --target jobs
build demandops-streamlit Dockerfile.streamlit
build demandops-airflow   Dockerfile.airflow
echo "pushed ${REGISTRY}/*:${TAG}"
