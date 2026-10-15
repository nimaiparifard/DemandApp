# ADR 0003 - SeaweedFS as the S3 store (instead of MinIO)

**Status:** accepted (2026-10-15)

## Context
The brief specifies MinIO as the self-hosted S3 store. While building the stack, `docker pull minio/minio`
and `minio/mc` failed with "repository does not exist" on Docker Hub, and the Quay tags
(including the one pinned by the `minio` Helm chart) could not be resolved either. MinIO stopped publishing
community container images, so a fresh install cannot pull them.

## Decision
Use **SeaweedFS** (`chrislusf/seaweedfs:4.48`, master + volume + filer + S3 gateway in one container/pod).
All code talks the plain S3 API through `pyarrow.fs.S3FileSystem` (storage layer), `boto3` (MLflow server) and
barman (CloudNativePG backups), so the store is a deployment detail.

Verified: pyarrow read/write/list, MLflow artifact upload/download through the artifact proxy, the simulator
landing bucket and the feature table all work against SeaweedFS in `docker compose`.

## Consequences
* Swapping to MinIO built from source, Garage, Ceph RGW, ArvanCloud Object Storage or AWS S3 = changing
  `S3_ENDPOINT_URL` and credentials.
* Not verified against SeaweedFS: CloudNativePG WAL archiving and Feast's S3 file source (both only render/validate).
* Credentials default to demo values; real ones go in SealedSecrets (`k8s-addons/secrets.example.yaml`).
