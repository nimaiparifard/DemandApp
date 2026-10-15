.DEFAULT_GOAL := help
SHELL := /bin/bash
COMPOSE := docker compose -f infra/docker-compose.yml
JOBS := $(COMPOSE) --profile jobs run --rm jobs
N ?= 300
ENV ?= local

help: ## list targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-18s %s\n", $$1, $$2}'

# ---------------------------------------------------------------- dev setup
setup: ## install python deps (editable) + git hooks
	pip install -e ".[dev,serving,statsforecast,app]"
	pre-commit install || true

data: ## put the M5 CSVs under DVC (remote = S3 store); download them from Kaggle first
	@test -f data/sales_train_evaluation.csv || (echo "Download M5 first: kaggle competitions download -c m5-forecasting-accuracy -p data && unzip -o data/*.zip -d data" && exit 1)
	dvc add data/calendar.csv data/sales_train_evaluation.csv data/sell_prices.csv
	dvc push

synthetic: ## small synthetic data set (no Kaggle needed) in data_synth/
	python scripts/make_synthetic_m5.py --out data_synth

# ---------------------------------------------------------------- ML (local, no containers)
train: ## backtest naive/ETS/LightGBM, log to MLflow (sqlite), register challenger. N=series
	demandops train --n-series $(N)

evaluate: ## same backtest, results table only (no baselines skipped)
	demandops train --n-series $(N)

retrain: ## train -> promotion gates -> promote (what DAG 3 does)
	demandops retrain --n-series $(N)

# ---------------------------------------------------------------- quality
lint: ## ruff + mypy
	ruff check . && ruff format --check . && mypy

fmt: ## auto-format
	ruff check --fix . && ruff format .

test: ## unit + integration tests (synthetic data, no services needed)
	pytest --cov --cov-report=term-missing --cov-fail-under=80

test-unit: ## fast tests only
	pytest tests/unit

load-test: ## k6: RPS and p95 at the current pod count (API_URL=http://localhost:8001)
	k6 run tests/load/infer.js

# ---------------------------------------------------------------- docker compose stack (no k8s)
up: ## start postgres/redis/s3/mlflow/serving/streamlit/prometheus/grafana
	$(COMPOSE) up -d --build

down: ## stop the stack (keeps volumes)
	$(COMPOSE) down

logs: ## follow logs
	$(COMPOSE) logs -f --tail=100

demo: ## run the whole pipeline in containers: simulate -> train/gates -> features -> batch -> API
	$(JOBS) simulate --days 3 --n-series $(N)
	$(JOBS) retrain --n-series $(N)
	$(JOBS) publish-features --n-series $(N)
	$(JOBS) batch-forecast
	$(COMPOSE) up -d --build serving streamlit
	@echo "API: http://localhost:8001/docs   UI: http://localhost:8501   MLflow: http://localhost:5000   Grafana: http://localhost:3001"

demo-video: ## re-record docs/media/demo.mp4 (needs the stack up + filled, Google Chrome, pip install playwright imageio-ffmpeg pillow)
	python scripts/record_demo.py --terminal scratch/terminal.txt --out docs/media/demo.mp4

# ---------------------------------------------------------------- kubernetes / IaC
helm-lint: ## lint + render the chart in every mode
	helm lint infra/helm/demandops
	helm template t infra/helm/demandops > /dev/null
	helm template t infra/helm/demandops -f infra/helm/demandops/values-vps.yaml > /dev/null
	helm template t infra/helm/demandops --set serving.mode=kserve > /dev/null

tofu-check: ## fmt + validate both OpenTofu environments
	tofu -chdir=infra/tofu fmt -recursive -check
	for e in local vps; do tofu -chdir=infra/tofu/envs/$$e init -backend=false -input=false && tofu -chdir=infra/tofu/envs/$$e validate; done

k3d-up: ## local Kubernetes + Argo CD (installs the whole platform from git)
	cd infra/tofu/envs/local && tofu init && tofu apply -auto-approve

k3d-down: ## destroy the local cluster
	cd infra/tofu/envs/local && tofu destroy -auto-approve

vps-up: ## provision the VPS (OpenTofu) then configure it (Ansible)
	cd infra/tofu/envs/vps && tofu init && tofu apply
	cd infra/ansible && ansible-galaxy collection install -r requirements.yml && ansible-playbook playbook.yml

images: ## build and push the 4 images to REGISTRY (default: local k3d registry)
	REGISTRY=$${REGISTRY:-k3d-demandops-registry.localhost:5111/demandops} TAG=$${TAG:-latest} ./scripts/build_push.sh

.PHONY: demo-video help setup data synthetic train evaluate retrain lint fmt test test-unit load-test up down logs demo helm-lint tofu-check k3d-up k3d-down vps-up images
