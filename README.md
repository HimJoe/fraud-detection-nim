# NVIDIA Financial Fraud Detection NIM — Enhanced Blueprint

A production-ready, containerized fraud detection pipeline built on the [NVIDIA Financial Fraud Detection Blueprint](https://build.nvidia.com/nvidia/financial-fraud-detection), extended with:

- **Synthetic transaction generation** (Faker + Kaggle seed data)
- **Feature Bus (FB)** shared Redis Streams for inter-module comms
- **GPU-accelerated inference** via NVIDIA Triton + RAPIDS cuDF
- **Real-time dashboard** with stress-test mode
- **Kubernetes-native** deployment on L40S nodes

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                        Feature Bus (Redis Streams)                    │
│   cpu_tx ──► cpu_features ──► cpu_scores                             │
│   gpu_tx ──────────────────► gpu_scores                              │
└──────────────────────────────────────────────────────────────────────┘
        ▲              ▲                    ▲
  [Generator]   [Data Prep / FE]     [Inference / Scoring]
  Faker + seed   cuDF / pandas        Triton + XGBoost/GNN
                                           │
                                    [Dashboard]
                               Metrics · Stress Mode · $$ Saved
```

### 4 Modules

| Module | Stream Reads | Stream Writes | Tech |
|--------|-------------|---------------|------|
| **generator** | — | `cpu_tx`, `gpu_tx` | Python, Faker, pandas |
| **data-prep** | `cpu_tx` | `cpu_features` | cuDF / pandas fallback |
| **inference** | `cpu_features`, `gpu_tx` | `cpu_scores`, `gpu_scores` | Triton gRPC, XGBoost |
| **dashboard** | `*_scores`, metrics | — | FastAPI + React |

---

## Quick Start (Docker Compose — Dev)

```bash
git clone <your-repo>
cd fraud-detection-nim

# Copy env template
cp .env.example .env

# Start everything (CPU-only dev mode)
docker compose up --build

# Dashboard → http://localhost:8080
```

## Production (Kubernetes on L40S)

```bash
# 1. Build & push all images
./scripts/build-push.sh <REGISTRY> <TAG>

# 2. Deploy to cluster
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/redis.yaml
kubectl apply -f k8s/

# 3. Watch rollout
kubectl -n fraud-detection rollout status deployment
```

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `REDIS_URL` | `redis://redis:6379` | Feature Bus endpoint |
| `TX_RATE_NORMAL` | `100` | Transactions/sec normal mode |
| `TX_RATE_STRESS` | `2000` | Transactions/sec stress mode |
| `TRITON_URL` | `triton:8001` | Triton gRPC endpoint |
| `KAGGLE_DATASET_PATH` | `/data/kaggle_seed.csv` | Seed data path |
| `GPU_ENABLED` | `true` | Enable GPU pipeline |
| `FRAUD_THRESHOLD` | `0.7` | Score threshold for flagging |

---

## Module Details

### 1. Generator (`modules/generator`)
- Loads Kaggle credit card fraud dataset as seed (statistical profiles)
- Uses Faker to synthesize realistic card/merchant/user data
- Writes to **two** Feature Bus streams simultaneously:
  - `cpu_tx` — for CPU data-prep pipeline
  - `gpu_tx` — for direct GPU inference pipeline
- Supports **STRESS MODE** (bumped via Redis pub/sub signal)

### 2. Data Prep (`modules/data-prep`)
- Reads `cpu_tx` stream
- Feature engineering: time deltas, velocity features, merchant risk scores, amount normalization
- Writes enriched features to `cpu_features` stream
- Uses cuDF when GPU available, pandas fallback for CPU nodes

### 3. Inference (`modules/inference`)
- Dual readers: `cpu_features` and `gpu_tx`
- Calls Triton Inference Server (GNN + XGBoost pipeline)
- Computes fraud scores + Shapley explainability values
- Writes results to `cpu_scores` / `gpu_scores` streams
- Exposes Prometheus metrics endpoint

### 4. Dashboard (`modules/dashboard`)
- FastAPI backend aggregating all stream metrics
- React frontend with real-time charts (WebSocket)
- Key metrics: $$ saved, tx throughput, CPU/GPU utilization, FB latency
- **Stress button** triggers generator rate spike via Redis pub/sub

---

## Kaggle Dataset

Download the [Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place at:

```
shared/data/kaggle_seed.csv
```

Or set `KAGGLE_DATASET_PATH` env var.

---

## Stress Test

Click **"⚡ Stress Test"** in the dashboard, or trigger via API:

```bash
curl -X POST http://localhost:8080/api/stress/enable
curl -X POST http://localhost:8080/api/stress/disable
```

This publishes a Redis pub/sub message → generator scales TX rate from `TX_RATE_NORMAL` → `TX_RATE_STRESS`.

---

## Repository Structure

```
fraud-detection-nim/
├── modules/
│   ├── generator/          # Synthetic TX generation
│   ├── data-prep/          # Feature engineering
│   ├── inference/          # Triton scoring
│   └── dashboard/          # Metrics UI + API
├── k8s/                    # Kubernetes manifests
├── scripts/                # Build, push, deploy helpers
├── shared/                 # Shared schemas, utils
├── docker-compose.yml      # Local dev stack
└── .env.example
```

---

## License

Built on [NVIDIA Financial Fraud Detection Blueprint](https://github.com/NVIDIA-AI-Blueprints/Financial-Fraud-Detection) — governed by [NVIDIA AI Foundation Models Community License](https://docs.nvidia.com/ai-foundation-models-community-license.pdf).
