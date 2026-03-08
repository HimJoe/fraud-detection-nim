"""
modules/dashboard/app.py
FastAPI backend for the fraud detection dashboard.
- REST endpoints for metrics snapshots
- WebSocket for real-time streaming metrics
- Stress test control endpoints
- Business value ($$  saved) calculation
"""
import os
import sys
import time
import json
import asyncio
import logging
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import redis.asyncio as aioredis

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from shared.featurebus.client import (
    STREAM_CPU_SCORES, STREAM_GPU_SCORES, STREAM_METRICS,
    STREAM_CPU_TX, STREAM_GPU_TX, STREAM_CPU_FEATURES,
    PUBSUB_CONTROL
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [dashboard] %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ─── Config ───────────────────────────────────────────────────────────────────
REDIS_URL         = os.getenv("REDIS_URL", "redis://redis:6379")
AVG_FRAUD_LOSS_USD = float(os.getenv("AVG_FRAUD_LOSS_USD", "175"))   # industry avg per fraud tx
FALSE_POSITIVE_COST = float(os.getenv("FALSE_POSITIVE_COST", "4.5"))  # ops cost per FP review

# ─── State accumulator ────────────────────────────────────────────────────────

class MetricsState:
    def __init__(self):
        self.reset_time          = time.time()
        self.total_tx            = 0
        self.cpu_tx              = 0
        self.gpu_tx              = 0
        self.cpu_fraud_count     = 0
        self.gpu_fraud_count     = 0
        self.cpu_fraud_value     = 0.0
        self.gpu_fraud_value     = 0.0
        self.cpu_tps             = 0.0
        self.gpu_tps             = 0.0
        self.cpu_latency_ms      = 0.0
        self.gpu_latency_ms      = 0.0
        self.dataprep_tps        = 0.0
        self.dataprep_latency_ms = 0.0
        self.generator_tps       = 0.0
        self.stress_mode         = False
        self.fb_stream_lengths   = {}
        self.last_update         = time.time()

    @property
    def total_fraud_count(self):
        return self.cpu_fraud_count + self.gpu_fraud_count

    @property
    def total_fraud_value(self):
        return self.cpu_fraud_value + self.gpu_fraud_value

    @property
    def dollars_saved(self):
        """Estimated $ saved = (frauds caught × avg loss) − false positive cost"""
        # Assume 85% detection rate, 5% FP rate for demo
        detection_rate   = 0.85
        fp_rate          = 0.05
        caught           = self.total_fraud_count * detection_rate
        fps              = self.total_tx * fp_rate * 0.01  # FP reviews are rare
        return max(0, caught * AVG_FRAUD_LOSS_USD - fps * FALSE_POSITIVE_COST)

    def to_dict(self):
        return {
            "timestamp":        time.time(),
            "stress_mode":      self.stress_mode,
            "total_tx":         self.total_tx,
            "cpu_tx":           self.cpu_tx,
            "gpu_tx":           self.gpu_tx,
            "total_fraud_count": self.total_fraud_count,
            "total_fraud_value": round(self.total_fraud_value, 2),
            "dollars_saved":    round(self.dollars_saved, 2),
            "cpu_tps":          round(self.cpu_tps, 1),
            "gpu_tps":          round(self.gpu_tps, 1),
            "generator_tps":    round(self.generator_tps, 1),
            "dataprep_tps":     round(self.dataprep_tps, 1),
            "cpu_latency_ms":   round(self.cpu_latency_ms, 2),
            "gpu_latency_ms":   round(self.gpu_latency_ms, 2),
            "dataprep_latency_ms": round(self.dataprep_latency_ms, 2),
            "fb_stream_lengths": self.fb_stream_lengths,
            "uptime_seconds":   round(time.time() - self.reset_time, 0),
        }


state = MetricsState()
ws_clients: list[WebSocket] = []


# ─── Background tasks ─────────────────────────────────────────────────────────

async def consume_metrics(r: aioredis.Redis):
    """Tail the metrics stream and update state."""
    last_id = "$"
    while True:
        try:
            results = await r.xread({STREAM_METRICS: last_id}, count=200, block=500)
            for _stream, messages in (results or []):
                for msg_id, data in messages:
                    last_id = msg_id
                    name    = data.get("name", "")
                    value   = float(data.get("value", 0))
                    labels  = json.loads(data.get("labels", "{}"))
                    pipeline = labels.get("pipeline", "")

                    if name == "generator_tps":
                        state.generator_tps = value
                        state.stress_mode   = labels.get("stress") == "True"
                    elif name == "inference_tps":
                        if pipeline == "cpu":  state.cpu_tps = value
                        else:                  state.gpu_tps = value
                    elif name == "inference_fraud_count":
                        if pipeline == "cpu":  state.cpu_fraud_count += int(value)
                        else:                  state.gpu_fraud_count += int(value)
                    elif name == "inference_fraud_value_usd":
                        if pipeline == "cpu":  state.cpu_fraud_value += value
                        else:                  state.gpu_fraud_value += value
                    elif name == "inference_latency_ms":
                        if pipeline == "cpu":  state.cpu_latency_ms = value
                        else:                  state.gpu_latency_ms = value
                    elif name == "dataprep_tps":
                        state.dataprep_tps = value
                    elif name == "dataprep_batch_latency_ms":
                        state.dataprep_latency_ms = value
                    elif name == "generator_tx_total":
                        state.total_tx = int(value)

        except Exception as e:
            logger.warning(f"Metrics consumer error: {e}")
            await asyncio.sleep(1)


async def poll_stream_lengths(r: aioredis.Redis):
    """Poll Feature Bus stream lengths for FB utilisation metrics."""
    streams = [STREAM_CPU_TX, STREAM_GPU_TX, STREAM_CPU_FEATURES,
               STREAM_CPU_SCORES, STREAM_GPU_SCORES]
    while True:
        try:
            lengths = {}
            for s in streams:
                try:
                    lengths[s] = await r.xlen(s)
                except Exception:
                    lengths[s] = 0
            state.fb_stream_lengths = lengths
        except Exception as e:
            logger.warning(f"Stream poll error: {e}")
        await asyncio.sleep(2)


async def broadcast_ws():
    """Push metrics to all connected WebSocket clients every second."""
    while True:
        if ws_clients:
            payload = json.dumps(state.to_dict())
            dead = []
            for ws in ws_clients:
                try:
                    await ws.send_text(payload)
                except Exception:
                    dead.append(ws)
            for ws in dead:
                ws_clients.remove(ws)
        await asyncio.sleep(1)


# ─── App lifecycle ────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    r = aioredis.from_url(REDIS_URL, decode_responses=True)
    app.state.redis = r
    asyncio.create_task(consume_metrics(r))
    asyncio.create_task(poll_stream_lengths(r))
    asyncio.create_task(broadcast_ws())
    logger.info("Dashboard started")
    yield
    await r.aclose()


app = FastAPI(title="Fraud Detection Dashboard API", lifespan=lifespan)


# ─── REST endpoints ───────────────────────────────────────────────────────────

@app.get("/api/metrics")
async def get_metrics():
    return state.to_dict()


@app.get("/api/health")
async def health():
    try:
        await app.state.redis.ping()
        return {"status": "ok", "redis": "connected"}
    except Exception:
        return {"status": "degraded", "redis": "disconnected"}


@app.post("/api/stress/enable")
async def stress_enable():
    await app.state.redis.publish(PUBSUB_CONTROL,
                                  json.dumps({"action": "stress_enable"}))
    state.stress_mode = True
    return {"stress": True}


@app.post("/api/stress/disable")
async def stress_disable():
    await app.state.redis.publish(PUBSUB_CONTROL,
                                  json.dumps({"action": "stress_disable"}))
    state.stress_mode = False
    return {"stress": False}


@app.post("/api/metrics/reset")
async def reset_metrics():
    state.__init__()
    return {"reset": True}


# ─── WebSocket ────────────────────────────────────────────────────────────────

@app.websocket("/ws/metrics")
async def ws_metrics(websocket: WebSocket):
    await websocket.accept()
    ws_clients.append(websocket)
    try:
        while True:
            await websocket.receive_text()   # keep-alive pings from client
    except WebSocketDisconnect:
        ws_clients.remove(websocket)


# ─── Static files (React build) ───────────────────────────────────────────────

STATIC_DIR = Path(__file__).parent / "static"
if STATIC_DIR.exists():
    app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
