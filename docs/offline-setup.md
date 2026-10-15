# Working with restricted internet (air-gapped style setup guide)

Goal: after one-time downloads, nothing in the pipeline needs the public internet.

## Week 0 checklist (do first - image pulls are the #1 time sink)

1. **Docker daemon mirror.** Copy `infra/mirrors/docker-daemon.json` to Docker's config (Docker Desktop ->
   Settings -> Docker Engine; Linux: `/etc/docker/daemon.json`). `docker.arvancloud.ir` answered `401` on `/v2/`
   when checked (= alive, auth challenge); mirrors come and go, so test with `curl -I https://<mirror>/v2/`.
2. **Python packages.** Put `infra/mirrors/pip.conf` (or `uv.toml`) in place. The sample mirror returned `503` when
   checked - pick any live mirror or run your own (`devpi`, Nexus) and keep PyPI as `extra-index-url` via a proxy.
   Build images with `--build-arg PIP_INDEX_URL=...`.
3. **Harbor with proxy-cache projects** (inside the cluster): `scripts/harbor_setup.sh` creates `proxy-dockerhub`,
   `proxy-ghcr`, `proxy-quay`, `proxy-k8s`. `infra/mirrors/registries.yaml` makes k3s/k3d pull through them, so each
   upstream image is downloaded once and then served locally.
4. **Helm charts.** Argo CD fetches charts from their public repositories. To remove that dependency run once
   `helm pull <chart> --version <v>` for every Application in `infra/argocd/apps`, `helm push` the `.tgz` files to
   Harbor (it is an OCI registry) and change the `repoURL` to `oci://harbor.../charts`. The versions are pinned
   in each Application, so what you pull is what the repo expects.
5. **Datasets.** Download M5 once (VPN/proxy allowed, or the copy linked from the official M5 repository), then
   `make data` (`dvc add` + `dvc push` to the S3 store). After that the pipeline never needs the internet for data.
   ```bash
   kaggle competitions download -c m5-forecasting-accuracy -p data && unzip -o data/m5-forecasting-accuracy.zip -d data
   ```
6. **k3s binary / Helm / OpenTofu providers.** `get.k3s.io`, GitHub releases and the HashiCorp registry can be
   blocked. Download the k3s binary once and set `k3s_binary_local`; OpenTofu uses its own registry (the Arvan and
   hashicorp providers installed fine from it in testing).

## Things verified vs not (as of this repo state)

| Item | Status |
|---|---|
| Docker Hub: `python:3.12-slim`, `postgres`, `redis`, `grafana`, `prom/prometheus` pull | worked from the dev machine |
| Docker Hub `minio/minio`, `minio/mc` | **gone** - see ADR 0003 |
| Helm repos used by the Applications | all reachable (HTTP 200) when checked |
| OpenTofu providers (arvancloud/arvan, hashicorp/helm, hashicorp/local) | installed and `validate` passes |
| Iranian PyPI/Docker mirrors | not relied upon; examples only |
