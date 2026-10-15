#!/usr/bin/env bash
# Create the Harbor projects: "demandops" (our images) + proxy-cache projects for upstream registries,
# so every external image is downloaded ONCE and served from Harbor afterwards.
#   HARBOR_URL=http://harbor.demandops.local HARBOR_PASSWORD=... ./scripts/harbor_setup.sh
set -euo pipefail
URL="${HARBOR_URL:-http://harbor.demandops.local}"
AUTH="admin:${HARBOR_PASSWORD:?set HARBOR_PASSWORD}"

api() { curl -fsS -u "$AUTH" -H 'Content-Type: application/json' "$@"; }

registry() { # name endpoint_url type
  api "$URL/api/v2.0/registries" -d "{\"name\":\"$1\",\"url\":\"$2\",\"type\":\"$3\",\"insecure\":false}" 2>/dev/null || true
  api "$URL/api/v2.0/registries" | python3 -c "import sys,json;print(next(r['id'] for r in json.load(sys.stdin) if r['name']=='$1'))"
}

project() { # name [registry_id]
  local body="{\"project_name\":\"$1\",\"public\":true"
  [ -n "${2:-}" ] && body="$body,\"registry_id\":$2"
  api "$URL/api/v2.0/projects" -d "$body}" 2>/dev/null && echo "created project $1" || echo "project $1 exists"
}

project demandops
project proxy-dockerhub "$(registry dockerhub https://hub.docker.com docker-hub)"
project proxy-ghcr      "$(registry ghcr https://ghcr.io github-ghcr)"
project proxy-quay      "$(registry quay https://quay.io quay)"
project proxy-k8s       "$(registry k8s https://registry.k8s.io docker-registry)"
echo "done. Point k3s at them with infra/mirrors/registries.yaml"
