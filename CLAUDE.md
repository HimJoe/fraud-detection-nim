# CLAUDE.md — Fraud Detection NIM

This file is read automatically by Claude Code on every session start.
It contains everything you need to contribute to this project without re-explaining the architecture.

---

## What this project is

A production-ready, containerised **financial fraud detection pipeline** built on the
[NVIDIA Financial Fraud Detection Blueprint](https://build.nvidia.com/nvidia/financial-fraud-detection).

It runs on a **lab VM with Docker**, images are pushed to a **local private registry**,
and deployed to **Kubernetes on L40S GPU nodes** via manifests in `k8s/`.

---

## Architecture — read this first

```
Kaggle CSV seed
     │
     ▼
┌─────────────┐   cpu_tx stream   ┌─────────────┐   cpu_features stream   ┌─────────────┐
│  generator  │ ────────────────► │  data-prep  │ ──────────────────────► │  inference  │
│  (module 1) │                   │  (module 2) │                         │  (module 3) │
│             │   gpu_tx stream   └─────────────┘                         │             │
│             │ ──────────────────────────────────────────────────────── ►│             │
└─────────────┘                                                            └─────────────┘
                                                                                  │
                                                          cpu_scores / gpu_scores │
                                                                                  ▼
                                                                        ┌─────────────────┐
                                                                        │    dashboard    │
                                                                        │   (module 4)    │
                                                                        │  FastAPI + React│
                                                                        └─────────────────┘
```

**The Feature Bus is Redis Streams** — all inter-module communication goes through it.
Never have modules call each other directly. Always go through the Feature Bus.

---

## Module summary

| Module | Path | Reads from | Writes to | Key tech |
|--------|------|-----------|-----------|----------|
| Generator | `modules/generator/main.py` | Kaggle CSV (seed) | `cpu_tx`, `gpu_tx` | Faker, numpy |
| Data Prep | `modules/data-prep/main.py` | `cpu_tx` | `cpu_features` | cuDF (GPU) / pandas fallback |
| Inference | `modules/inference/main.py` | `cpu_features`, `gpu_tx` | `cpu_scores`, `gpu_scores` | Triton gRPC, XGBoost |
| Dashboard | `modules/dashboard/app.py` | `metrics` stream, all score streams | — | FastAPI, WebSocket, React |

---

## Shared code — always use this

**`shared/featurebus/client.py`** is the single source of truth for:
- Stream names (`STREAM_CPU_TX`, `STREAM_GPU_TX`, `STREAM_CPU_FEATURES`, etc.)
- Dataclass schemas (`RawTransaction`, `EnrichedFeatures`, `FraudScore`, `Metric`)
- `FeatureBusClient` — all Redis read/write operations

When adding a new stream or changing a schema field, **edit here first**, then update
the modules that use it. Do not hardcode stream names in individual modules.

---

## Stream names (from shared/featurebus/client.py)

```python
STREAM_CPU_TX       = "cpu_tx"
STREAM_GPU_TX       = "gpu_tx"
STREAM_CPU_FEATURES = "cpu_features"
STREAM_CPU_SCORES   = "cpu_scores"
STREAM_GPU_SCORES   = "gpu_scores"
STREAM_METRICS      = "metrics"
PUBSUB_CONTROL      = "control"   # stress mode / rate control signals
```

---

## Infrastructure

### Lab VM (your build machine)
- Has Docker installed
- Workflow: `git pull` → `./scripts/build-push.sh <REGISTRY> <TAG>` → images in local registry
- Local registry address goes in `.env` as `REGISTRY=`

### Kubernetes cluster
- GPU nodes: **L40S**, labelled `nvidia.com/gpu.product: NVIDIA-L40S`
- GPU modules (data-prep, inference, triton) have `nodeSelector` + `tolerations` in `k8s/deployments.yaml`
- CPU modules (generator, dashboard) run on any node
- Namespace: `fraud-detection`

### Deploy flow
```bash
./scripts/build-push.sh registry.local:5000 v1.0.0
./scripts/deploy.sh registry.local:5000 v1.0.0
```

### Local dev (no GPU needed)
```bash
docker compose up --build
# Dashboard → http://localhost:8080
```

---

## Key environment variables

| Variable | Default | Set in |
|----------|---------|--------|
| `REDIS_URL` | `redis://redis:6379` | `k8s/configmap.yaml` |
| `TX_RATE_NORMAL` | `100` (dev), `500` (k8s) | configmap |
| `TX_RATE_STRESS` | `2000` (dev), `5000` (k8s) | configmap |
| `GPU_ENABLED` | `false` (compose), `true` (k8s) | per-environment |
| `TRITON_URL` | `triton:8001` | configmap |
| `FRAUD_THRESHOLD` | `0.7` | configmap |
| `KAGGLE_DATASET_PATH` | `/data/kaggle_seed.csv` | configmap |

All k8s env vars live in `k8s/configmap.yaml`. Docker Compose env vars are in `docker-compose.yml` under `x-common-env`.

---

## Kaggle dataset

Download from: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
Place at: `shared/data/kaggle_seed.csv`

The generator loads this on startup and uses it as a statistical seed for realistic
amount distributions and fraud rates. If the file is missing, it falls back to
synthetic defaults (still works, just less realistic).

---

## Stress test / control signals

The generator listens on Redis pub/sub channel `control` for JSON messages:

```python
{"action": "stress_enable"}   # bumps TX rate to TX_RATE_STRESS
{"action": "stress_disable"}  # returns to TX_RATE_NORMAL
{"action": "set_rate", "rate": 750}  # set arbitrary rate
```

The dashboard **Stress Test button** publishes these via `POST /api/stress/enable` and
`POST /api/stress/disable`. You can also trigger from the CLI:

```bash
redis-cli publish control '{"action":"stress_enable"}'
```

---

## Triton Inference Server

- Used for production GPU inference (GNN + XGBoost ensemble)
- gRPC endpoint: `triton:8001`
- Model name: `fraud_ensemble`
- Model repository should be mounted at `/models` in the Triton container (see `k8s/deployments.yaml`)
- **Fallback**: if Triton is unreachable, the inference module automatically falls back to a local heuristic scorer — useful for dev/testing without a trained model

---

## Adding a new feature — checklist

1. If adding a new field to a stream: update the dataclass in `shared/featurebus/client.py`
2. Update the writing module (e.g. generator for `RawTransaction`)
3. Update the reading module (e.g. data-prep for `RawTransaction`)
4. If it's a new metric: add to `STREAM_METRICS` writes and update `modules/dashboard/app.py` to consume it
5. Update `k8s/configmap.yaml` if it needs a new env var
6. Test locally with `docker compose up --build` before pushing

---

## Common tasks for Claude Code

**Run the stack locally:**
```bash
docker compose up --build
docker compose logs -f generator
docker compose logs -f inference
```

**Check Feature Bus stream depths:**
```bash
redis-cli xlen cpu_tx
redis-cli xlen cpu_features
redis-cli xlen gpu_scores
```

**Watch metrics in real time:**
```bash
redis-cli xread COUNT 10 STREAMS metrics 0
```

**Trigger stress mode manually:**
```bash
redis-cli publish control '{"action":"stress_enable"}'
sleep 30
redis-cli publish control '{"action":"stress_disable"}'
```

**Build and push a single module (faster iteration):**
```bash
DOCKER_BUILDKIT=1 docker build -f modules/inference/Dockerfile --target cpu \
  -t registry.local:5000/fraud-inference:dev . && \
  docker push registry.local:5000/fraud-inference:dev
kubectl -n fraud-detection rollout restart deployment/inference
```

---

## Things to know / watch out for

- **Data-prep uses consumer groups** — if you restart it, it will re-read from where it left off (not from the beginning). To reset: `redis-cli xgroup setid cpu_tx data-prep-group 0`
- **GPU vs CPU pipeline**: transactions are split by `GPU_TX_FRACTION` (default 50%). GPU path skips data-prep entirely and goes direct to inference. CPU path goes through data-prep first.
- **Dashboard dollars saved** is an estimate: `(fraud_caught × $175 avg loss) − (false_positive_reviews × $4.50)`. Tune `AVG_FRAUD_LOSS_USD` and `FALSE_POSITIVE_COST` in configmap to match your actual numbers.
- **Triton model files are not in this repo** — they are produced by the NVIDIA Financial Fraud Training container. See the [NVIDIA blueprint](https://github.com/NVIDIA-AI-Blueprints/Financial-Fraud-Detection) for model building steps.
- **HPA** will auto-scale data-prep and inference between 1–4 replicas based on CPU utilisation (thresholds in `k8s/hpa.yaml`). During stress tests you should see this kick in.

---

## File map

```
fraud-detection-nim/
├── CLAUDE.md                        ← you are here
├── README.md                        ← team-facing docs
├── docker-compose.yml               ← local dev stack
├── .env.example                     ← copy to .env and fill in
├── shared/
│   └── featurebus/
│       └── client.py                ← THE shared schema + FB client
├── modules/
│   ├── generator/
│   │   ├── main.py                  ← TX generation loop
│   │   ├── Dockerfile
│   │   └── requirements.txt
│   ├── data-prep/
│   │   ├── main.py                  ← feature engineering
│   │   └── Dockerfile               ← multi-stage: gpu (default) / cpu
│   ├── inference/
│   │   ├── main.py                  ← Triton client + fallback scorer
│   │   └── Dockerfile               ← multi-stage: gpu (default) / cpu
│   └── dashboard/
│       ├── app.py                   ← FastAPI backend + WebSocket
│       ├── Dockerfile
│       └── static/
│           └── index.html           ← React frontend (single file)
├── k8s/
│   ├── namespace.yaml
│   ├── redis.yaml                   ← Feature Bus
│   ├── configmap.yaml               ← all env vars
│   ├── deployments.yaml             ← all 4 modules + Triton
│   ├── hpa.yaml                     ← auto-scaling
│   └── pvcs.yaml                    ← data + model storage
└── scripts/
    ├── build-push.sh                ← build all images → registry
    └── deploy.sh                    ← kubectl apply everything
```
