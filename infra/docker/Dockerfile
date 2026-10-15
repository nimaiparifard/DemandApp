# syntax=docker/dockerfile:1
# Multi-stage image for everything that runs demandops code: serving API and batch jobs.
#   docker build -f infra/docker/Dockerfile --target serving -t demandops-serving .
#   docker build -f infra/docker/Dockerfile --target jobs    -t demandops-jobs .
# Behind a mirror (Iran): --build-arg PIP_INDEX_URL=https://<your-mirror>/simple
ARG PYTHON_IMAGE=python:3.12-slim

FROM ${PYTHON_IMAGE} AS base
ARG PIP_INDEX_URL=https://pypi.org/simple
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_INDEX_URL=${PIP_INDEX_URL}
# libgomp1: OpenMP runtime required by LightGBM
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
# dependencies first (cached until pyproject.toml changes), then the code
COPY pyproject.toml README.md ./
RUN mkdir -p src/demandops && touch src/demandops/__init__.py \
    && pip install ".[serving,postgres,s3]" && pip uninstall -y demandops
COPY src ./src
RUN pip install --no-deps .
COPY pipelines ./pipelines
ENV PYTHONPATH=/app
# non-root
RUN useradd --uid 10001 --create-home app && mkdir -p /data /app/mlruns && chown -R app /app /data
USER 10001

# ---- online inference (KServe custom container / Argo Rollout pods) ----
FROM base AS serving
EXPOSE 8001
HEALTHCHECK --interval=15s --timeout=3s --start-period=60s \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8001/health').status==200 else 1)"
CMD ["uvicorn", "demandops.serving.app:app", "--host", "0.0.0.0", "--port", "8001", "--workers", "1"]

# ---- batch / training / simulator jobs (CronJobs, compose `jobs` profile) ----
FROM base AS jobs
USER root
RUN mkdir -p /tmp/stub/src/demandops && touch /tmp/stub/src/demandops/__init__.py \
    && cp pyproject.toml README.md /tmp/stub/ \
    && (cd /tmp/stub && (pip install ".[statsforecast,monitoring]" || pip install ".[statsforecast]") \
        && pip uninstall -y demandops) \
    && rm -rf /tmp/stub
RUN pip install --no-deps --force-reinstall .
COPY src/demandops/feature_repo /app/feature_repo
USER 10001
ENTRYPOINT ["demandops"]
CMD ["--help"]
