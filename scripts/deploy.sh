#!/usr/bin/env bash
# scripts/deploy.sh
# Deploy or update all K8s resources.
# Substitutes REGISTRY and TAG into deployment manifests before applying.
#
# Usage:
#   ./scripts/deploy.sh <REGISTRY> <TAG>
#
set -euo pipefail

REGISTRY=${1:?"Usage: $0 <REGISTRY> <TAG>"}
TAG=${2:?"Usage: $0 <REGISTRY> <TAG>"}

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
K8S="${ROOT}/k8s"

echo "🚀 Deploying fraud-detection-nim to Kubernetes"
echo "   REGISTRY=${REGISTRY}  TAG=${TAG}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Create namespace first
kubectl apply -f "${K8S}/namespace.yaml"

# Apply infra (Redis / Feature Bus)
kubectl apply -f "${K8S}/pvcs.yaml"
kubectl apply -f "${K8S}/redis.yaml"
kubectl apply -f "${K8S}/configmap.yaml"

# Wait for Redis to be ready
echo "⏳ Waiting for Redis..."
kubectl -n fraud-detection wait --for=condition=available --timeout=60s deployment/redis

# Substitute registry and tag in deployments, apply via stdin
sed "s|\${REGISTRY}|${REGISTRY}|g; s|\${TAG}|${TAG}|g" \
  "${K8S}/deployments.yaml" | kubectl apply -f -

# Apply HPA
kubectl apply -f "${K8S}/hpa.yaml"

echo ""
echo "⏳ Waiting for rollouts..."
for dep in generator data-prep inference triton dashboard; do
  echo "  → ${dep}"
  kubectl -n fraud-detection rollout status deployment/${dep} --timeout=120s || true
done

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ Deployment complete!"
echo ""
echo "Dashboard:"
DASHBOARD_IP=$(kubectl -n fraud-detection get svc dashboard \
  -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || echo "<pending>")
echo "  http://${DASHBOARD_IP}"
echo ""
echo "Useful commands:"
echo "  kubectl -n fraud-detection get pods"
echo "  kubectl -n fraud-detection logs -f deploy/generator"
echo "  kubectl -n fraud-detection logs -f deploy/inference"
echo "  kubectl -n fraud-detection top pods"
