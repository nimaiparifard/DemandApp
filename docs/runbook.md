# Runbook

Commands assume the repo root. `make help` lists every target.

## 1. First run on a laptop without Kubernetes (docker compose)

```bash
cp .env.example .env
# M5 data: see docs/offline-setup.md (Kaggle). No Kaggle? use synthetic data:
python scripts/make_synthetic_m5.py --out data --series 100 --days 500   # then: SIMULATOR_START_DAY=490
make up                       # postgres, redis, s3, mlflow, serving, streamlit, prometheus, grafana
make demo N=300               # simulate 3 days -> train + gates -> features -> batch forecast -> API
```

| What | URL |
|---|---|
| API docs | http://localhost:8001/docs |
| Streamlit | http://localhost:8501 (`STREAMLIT_PORT=8601 make up` if Windows reserved 8501) |
| MLflow | http://localhost:5000 |
| Grafana (admin/admin) | http://localhost:3001 -> folder DemandOps |
| Prometheus | http://localhost:9090 |

On Windows use `127.0.0.1`, not `localhost`, from Python clients: `localhost` resolves to IPv6 first and Docker
Desktop's port proxy can hang for 30 s before falling back.

## 2. Kubernetes (k3d on the laptop)

```bash
# prerequisites: docker, k3d, kubectl, helm, tofu (OpenTofu)
# 1. put YOUR repo URL into the Argo CD manifests
sed -i 's#YOUR_GITHUB_USER#<you>#' infra/argocd/root-app.yaml infra/argocd/apps/*.yaml
# 2. cluster + registry + Argo CD (Argo CD then installs everything from git)
make k3d-up
# 3. secrets: seal infra/k8s-addons/secrets.example.yaml (see the header of that file), commit, push
# 4. images
make images                                   # builds + pushes to the local k3d registry
# 5. M5 data into the PVC (once)
kubectl -n demandops run loader --image=busybox --restart=Never --overrides='{"spec":{"containers":[{"name":"loader","image":"busybox","command":["sleep","3600"],"volumeMounts":[{"name":"d","mountPath":"/data"}]}],"volumes":[{"name":"d","persistentVolumeClaim":{"claimName":"demandops-m5-data"}}]}}'
kubectl -n demandops cp data/calendar.csv loader:/data/ && \
kubectl -n demandops cp data/sales_train_evaluation.csv loader:/data/ && \
kubectl -n demandops cp data/sell_prices.csv loader:/data/
# 6. watch it converge
kubectl -n argocd get applications
```

First model: trigger the `retrain` DAG once in the Airflow UI (it registers the first `champion`), then run
`daily_ingest`, then `daily_batch_forecast`.

## 3. VPS (one-node k3s)

```bash
cp infra/tofu/envs/vps/terraform.tfvars.example infra/tofu/envs/vps/terraform.tfvars   # edit
make vps-up        # tofu apply (VM + security group + generated Ansible inventory) then the Ansible playbook
```
No cloud API? Create the VM by hand, write `infra/ansible/inventory.ini` yourself, run only the playbook.
`k3s_binary_local` in `group_vars/all.yml` lets you push a k3s binary downloaded once on another machine.

## 4. Operations

| Task | How |
|---|---|
| Force a retrain | Airflow UI -> `retrain` -> Trigger |
| Demo the drift loop | `kubectl -n demandops patch cronjob demandops-simulator -p '{"spec":{"jobTemplate":{"spec":{"template":{"spec":{"containers":[{"name":"simulator","args":["simulate","--days","1","--shift","3.0"]}]}}}}}}'` (4x demand) and wait for `monitoring` + alert |
| Roll back a model | `git revert` the `model.version` commit; Argo CD syncs the previous version (or `kubectl argo rollouts abort demand-forecaster`) |
| Watch a rollout | `kubectl argo rollouts get rollout demand-forecaster -n demandops -w` |
| Load test | `make load-test` (k6); run at 1 / 3 / 5 replicas and record the p95 in `docs/results.md` |
| Chaos-lite | while k6 runs: `kubectl delete pod -n demandops -l app=demand-forecaster --field-selector=status.phase=Running --wait=false` (one pod) and check `http_req_failed` |

## 5. Backup and restore

* Postgres: CloudNativePG `ScheduledBackup` (02:00) + continuous WAL to bucket `pg-backups`.
  Restore into a new cluster by adding to `postgres/cluster.yaml`:
  `bootstrap: { recovery: { source: demandops-pg } }` with `externalClusters` pointing at the same barman store.
* Object store: `mc mirror`/`rclone` the buckets to a second disk or VPS (script it as a CronJob).
* k3s state + sealed-secrets keys: Ansible `backup` role (cron, 14 days).
* **A restore has not been rehearsed on this repo yet** - do it once and record the date here before relying on it.

## 6. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `serving` restarts at first start | no `champion` model yet; run `make demo` / trigger `retrain` |
| `daily_ingest` fails with `DataQualityError` | by design: a check failed; message lists which (negative sales, missing day, gaps) |
| Retrain ends "kept champion" | read the reasons in the e-mail / task log (gate thresholds are in `pipelines/gates.py`) |
| Rollout aborted | `kubectl argo rollouts get rollout ...`; the failing AnalysisRun shows the Prometheus value |
| Images do not pull | see docs/offline-setup.md (mirrors, Harbor proxy cache) |
| `ports are not available ... 8501` (Windows) | port in a reserved Hyper-V range; set `STREAMLIT_PORT` |
