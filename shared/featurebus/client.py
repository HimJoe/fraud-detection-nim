"""
shared/featurebus/client.py
Feature Bus client — thin wrapper around Redis Streams.
All modules use this to read/write transactions and features.
"""
import json
import time
import logging
from dataclasses import dataclass, asdict
from typing import Optional, Iterator
import redis

logger = logging.getLogger(__name__)

# ─── Stream names ────────────────────────────────────────────────────────────
STREAM_CPU_TX           = "cpu_tx"
STREAM_GPU_TX           = "gpu_tx"
STREAM_CPU_FEATURES     = "cpu_features"
STREAM_CPU_PENDING      = "cpu_pending"   # in-flight: scored but not yet written
STREAM_GPU_PENDING      = "gpu_pending"   # in-flight: scored but not yet written
STREAM_CPU_SCORES       = "cpu_scores"
STREAM_GPU_SCORES       = "gpu_scores"
STREAM_METRICS          = "metrics"
PUBSUB_CONTROL          = "control"       # stress / config signals

# Ordered list for dashboard FB flow display
STREAM_FLOW_CPU = [STREAM_CPU_TX, STREAM_CPU_FEATURES, STREAM_CPU_PENDING, STREAM_CPU_SCORES]
STREAM_FLOW_GPU = [STREAM_GPU_TX, STREAM_GPU_PENDING, STREAM_GPU_SCORES]
ALL_STREAMS     = list(dict.fromkeys(STREAM_FLOW_CPU + STREAM_FLOW_GPU + [STREAM_METRICS]))

# ─── Schema dataclasses ───────────────────────────────────────────────────────

@dataclass
class RawTransaction:
    tx_id: str
    timestamp: float
    amount: float
    card_number: str          # tokenised / last4
    merchant_id: str
    merchant_category: str
    merchant_lat: float
    merchant_lon: float
    cardholder_id: str
    prev_tx_amount: float
    prev_tx_timestamp: float
    is_online: bool
    currency: str
    pipeline: str             # "cpu" | "gpu"

@dataclass
class EnrichedFeatures:
    tx_id: str
    timestamp: float
    pipeline: str
    # raw pass-throughs
    amount: float
    merchant_category: str
    is_online: bool
    # engineered
    amount_log: float
    time_since_last_tx: float          # seconds
    tx_velocity_1h: int                # tx count last hour (approx)
    amount_velocity_1h: float          # $ in last hour (approx)
    merchant_risk_score: float         # 0-1 static lookup
    distance_from_last_tx: float       # km
    hour_of_day: int
    day_of_week: int
    is_weekend: bool

@dataclass
class FraudScore:
    tx_id: str
    timestamp: float
    pipeline: str
    fraud_score: float          # 0-1
    is_fraud: bool
    amount: float
    shapley_top_feature: str
    shapley_top_value: float
    inference_latency_ms: float

@dataclass
class Metric:
    ts: float
    name: str
    value: float
    labels: dict              # e.g. {"pipeline": "gpu"}


# ─── Client ───────────────────────────────────────────────────────────────────

class FeatureBusClient:
    def __init__(self, redis_url: str = "redis://redis:6379", maxlen: int = 100_000):
        self.r = redis.from_url(redis_url, decode_responses=True)
        self.maxlen = maxlen

    # ── write helpers ────────────────────────────────────────────────────────

    def write_raw_tx(self, tx: RawTransaction):
        data = asdict(tx)
        data["is_online"] = int(data["is_online"])
        self.r.xadd(STREAM_CPU_TX if tx.pipeline == "cpu" else STREAM_GPU_TX,
                    data, maxlen=self.maxlen, approximate=True)

    def write_features(self, feat: EnrichedFeatures):
        data = asdict(feat)
        data["is_online"] = int(data["is_online"])
        data["is_weekend"] = int(data["is_weekend"])
        self.r.xadd(STREAM_CPU_FEATURES, data, maxlen=self.maxlen, approximate=True)

    def write_pending(self, tx_id: str, pipeline: str, batch_size: int = 1):
        """Write a pending marker to the FB before scoring — proves every score touches the bus."""
        stream = STREAM_CPU_PENDING if pipeline == "cpu" else STREAM_GPU_PENDING
        self.r.xadd(stream, {"tx_id": tx_id, "ts": time.time(),
                              "batch_size": batch_size, "pipeline": pipeline},
                    maxlen=50_000, approximate=True)

    def write_score(self, score: FraudScore):
        stream = STREAM_CPU_SCORES if score.pipeline == "cpu" else STREAM_GPU_SCORES
        self.r.xadd(stream, asdict(score), maxlen=self.maxlen, approximate=True)

    def write_metric(self, metric: Metric):
        data = {"ts": metric.ts, "name": metric.name,
                "value": metric.value, "labels": json.dumps(metric.labels)}
        self.r.xadd(STREAM_METRICS, data, maxlen=10_000, approximate=True)

    # ── read helpers ─────────────────────────────────────────────────────────

    def read_stream(self, stream: str, group: str, consumer: str,
                    count: int = 100, block_ms: int = 1000) -> list[dict]:
        """Read from a consumer group, auto-creating if needed."""
        try:
            self.r.xgroup_create(stream, group, id="0", mkstream=True)
        except redis.exceptions.ResponseError:
            pass   # group already exists

        results = self.r.xreadgroup(group, consumer, {stream: ">"}, count=count, block=block_ms)
        if not results:
            return []
        entries = []
        for _stream, messages in results:
            for msg_id, data in messages:
                data["_id"] = msg_id
                entries.append(data)
                self.r.xack(stream, group, msg_id)
        return entries

    def iter_stream(self, stream: str, group: str, consumer: str,
                    count: int = 100) -> Iterator[dict]:
        while True:
            for entry in self.read_stream(stream, group, consumer, count=count):
                yield entry

    # ── control pub/sub ───────────────────────────────────────────────────────

    def publish_control(self, message: dict):
        self.r.publish(PUBSUB_CONTROL, json.dumps(message))

    def subscribe_control(self):
        ps = self.r.pubsub(ignore_subscribe_messages=True)
        ps.subscribe(PUBSUB_CONTROL)
        return ps

    # ── metrics snapshot ─────────────────────────────────────────────────────

    def stream_len(self, stream: str) -> int:
        try:
            return self.r.xlen(stream)
        except Exception:
            return 0

    def stream_info(self, stream: str) -> dict:
        try:
            info = self.r.xinfo_stream(stream)
            return {"length": info["length"], "last_generated_id": info.get("last-generated-id", "")}
        except Exception:
            return {"length": 0}

    def ping(self) -> bool:
        try:
            return self.r.ping()
        except Exception:
            return False
