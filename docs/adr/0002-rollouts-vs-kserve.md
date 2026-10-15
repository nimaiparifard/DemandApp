# ADR 0002 - Argo Rollouts runs the canary; KServe is an alternative serving mode

**Status:** accepted

## Context
The brief asks for KServe (RawDeployment) **and** automatic canary releases with rollback driven by Argo Rollouts.
These do not compose: in RawDeployment mode KServe owns the Deployment it creates, so Argo Rollouts cannot manage
its pods, and KServe's own `canaryTrafficPercent` is only implemented for Serverless (Knative) mode, which needs
Knative + a service mesh (too heavy for one small VPS).

## Decision
* `serving.mode: rollout` (default): an Argo `Rollout` runs the inference image (it already speaks the KServe V2
  route `/v2/models/demand-forecaster/infer`). Traffic is split through ingress-nginx; an `AnalysisTemplate`
  queries Prometheus for the canary's 5xx rate and p95 (filtering by the `model_version` label) and aborts on failure.
* `serving.mode: kserve`: the same image as a KServe `InferenceService` (RawDeployment, custom container, HPA).
  Rolling updates only; no automated canary.
Both render and pass schema validation (`kubeconform` with the CRD catalogue).

## Consequences
* The canary/rollback requirement is met; the KServe requirement is met as a switchable mode, not simultaneously.
* If KServe's Serverless mode is ever adopted, the Rollout template can be dropped and `canaryTrafficPercent`
  used instead.
