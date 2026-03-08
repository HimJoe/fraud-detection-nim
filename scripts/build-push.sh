#!/usr/bin/env bash
# scripts/build-push.sh
# Build all module images and push to your local/private registry.
#
# Usage:
#   ./scripts/build-push.sh <REGISTRY> <TAG>
#   ./scripts/build-push.sh registry.local:5000 v1.0.0
#
set -euo pipefail

REGISTRY=${1:?"Usage: $0 <REGISTRY> <TAG>"}
TAG=${2:?"Usage: $0 <REGISTRY> <TAG>"}

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "🔨 Building fraud detection images → ${REGISTRY} tag=${TAG}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

build_push() {
  local module=$1
  local image_name=$2
  local target=${3:-""}       # optional multi-stage target

  echo ""
  echo "▶ Building: ${image_name}:${TAG}"
  cd "${ROOT}"

  ARGS=(
    "--build-arg" "BUILDKIT_INLINE_CACHE=1"
    "-f" "modules/${module}/Dockerfile"
    "-t" "${REGISTRY}/${image_name}:${TAG}"
    "-t" "${REGISTRY}/${image_name}:latest"
  )
  [[ -n "${target}" ]] && ARGS+=("--target" "${target}")
  ARGS+=(".")

  DOCKER_BUILDKIT=1 docker build "${ARGS[@]}"

  echo "▶ Pushing: ${image_name}:${TAG}"
  docker push "${REGISTRY}/${image_name}:${TAG}"
  docker push "${REGISTRY}/${image_name}:latest"
  echo "✅ ${image_name} done"
}

# Generator — CPU only
build_push "generator"  "fraud-generator"

# Data Prep — GPU (default target), or pass "cpu" for dev
build_push "data-prep"  "fraud-data-prep"  "${DATA_PREP_TARGET:-gpu}"

# Inference — GPU (default), Triton SDK image
build_push "inference"  "fraud-inference"  "${INFERENCE_TARGET:-gpu}"

# Dashboard — CPU only
build_push "dashboard"  "fraud-dashboard"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ All images built and pushed to ${REGISTRY}"
echo ""
echo "Next steps:"
echo "  1. Update k8s/deployments.yaml with your REGISTRY=${REGISTRY}"
echo "  2. kubectl apply -f k8s/"
echo "  3. kubectl -n fraud-detection rollout status deployment"
