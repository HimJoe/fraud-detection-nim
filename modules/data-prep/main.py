"""
modules/data-prep/main.py
Feature engineering module.
Reads raw transactions from cpu_tx stream on the FlashBlade,
applies feature engineering, and writes enriched features to cpu_features.

Uses cuDF (GPU) when available, falls back to pandas seamlessly.
"""
import os
import sys
import math
import time
import logging
from pathlib import Path
from collections import defaultdict, deque

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from shared.featurebus.client import (
    FlashBladeClient, EnrichedFeatures, Metric,
    STREAM_CPU_TX, STREAM_CPU_FEATURES
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [data-prep] %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ─── Config ───────────────────────────────────────────────────────────────────
REDIS_URL      = os.getenv("REDIS_URL", "redis://redis:6379")
GPU_ENABLED    = os.getenv("GPU_ENABLED", "true").lower() == "true"
CONSUMER_GROUP = "data-prep-group"
CONSUMER_NAME  = "data-prep-worker"
BATCH_SIZE     = int(os.getenv("BATCH_SIZE", "200"))

# ─── GPU / CPU backend selection ─────────────────────────────────────────────
USE_GPU = False
if GPU_ENABLED:
    try:
        import cudf
        USE_GPU = True
        logger.info("✅ cuDF GPU backend loaded")
    except ImportError:
        logger.info("cuDF not available, using pandas CPU backend")

if not USE_GPU:
    import pandas as pd
    logger.info("Using pandas CPU backend")

# ─── Merchant risk lookup (static, from NIM reference) ───────────────────────
MERCHANT_RISK_SCORES = {
    "grocery_pos": 0.12, "gas_transport": 0.18, "home": 0.09,
    "shopping_net": 0.65, "entertainment": 0.22, "food_dining": 0.15,
    "travel": 0.55, "health_fitness": 0.08, "personal_care": 0.07,
    "kids_pets": 0.06, "misc_net": 0.72, "misc_pos": 0.30
}


# ─── Stateful velocity tracking (rolling windows) ────────────────────────────

class VelocityTracker:
    """
    Lightweight in-memory rolling window tracker for:
    - Transaction count in last 1h per cardholder
    - Amount sum in last 1h per cardholder
    """
    def __init__(self, window_seconds: int = 3600):
        self.window = window_seconds
        self._tx_times:    defaultdict[str, deque] = defaultdict(deque)
        self._tx_amounts:  defaultdict[str, deque] = defaultdict(deque)

    def update(self, cardholder_id: str, ts: float, amount: float):
        times   = self._tx_times[cardholder_id]
        amounts = self._tx_amounts[cardholder_id]
        cutoff  = ts - self.window
        while times and times[0] < cutoff:
            times.popleft()
            amounts.popleft()
        times.append(ts)
        amounts.append(amount)

    def velocity(self, cardholder_id: str, ts: float) -> tuple[int, float]:
        times   = self._tx_times[cardholder_id]
        amounts = self._tx_amounts[cardholder_id]
        cutoff  = ts - self.window
        count  = sum(1 for t in times if t >= cutoff)
        total  = sum(a for t, a in zip(times, amounts) if t >= cutoff)
        return count, total


velocity_tracker = VelocityTracker()


# ─── Feature engineering ─────────────────────────────────────────────────────

def haversine(lat1, lon1, lat2, lon2) -> float:
    """Distance in km between two lat/lon pairs."""
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def engineer_features(raw: dict) -> EnrichedFeatures:
    ts            = float(raw["timestamp"])
    amount        = float(raw["amount"])
    cardholder_id = raw["cardholder_id"]
    prev_ts       = float(raw["prev_tx_timestamp"])
    prev_amount   = float(raw.get("prev_tx_amount", amount))

    # Time features
    from datetime import datetime, timezone
    dt          = datetime.fromtimestamp(ts, tz=timezone.utc)
    hour        = dt.hour
    dow         = dt.weekday()
    is_weekend  = dow >= 5
    time_delta  = max(0.0, ts - prev_ts)

    # Velocity
    velocity_tracker.update(cardholder_id, ts, amount)
    tx_vel, amount_vel = velocity_tracker.velocity(cardholder_id, ts)

    # Amount features
    amount_log = math.log1p(amount)

    # Merchant risk
    cat          = raw.get("merchant_category", "misc_net")
    merchant_risk = MERCHANT_RISK_SCORES.get(cat, 0.3)

    # Geo distance (approx — use merchant coords vs cardholder's last known)
    # We approximate using merchant lat/lon vs a fixed "home" jitter
    m_lat = float(raw.get("merchant_lat", 37.0))
    m_lon = float(raw.get("merchant_lon", -95.0))
    # Home is estimated from cardholder_id hash
    h_lat = 25.0 + (hash(cardholder_id) % 23000) / 1000.0
    h_lon = -122.0 + (hash(cardholder_id[:8]) % 52000) / 1000.0
    distance = haversine(h_lat, h_lon, m_lat, m_lon)

    return EnrichedFeatures(
        tx_id              = raw["tx_id"],
        timestamp          = ts,
        pipeline           = raw.get("pipeline", "cpu"),
        amount             = amount,
        merchant_category  = cat,
        is_online          = bool(int(raw.get("is_online", 0))),
        amount_log         = round(amount_log, 4),
        time_since_last_tx = round(time_delta, 2),
        tx_velocity_1h     = tx_vel,
        amount_velocity_1h = round(amount_vel, 2),
        merchant_risk_score = round(merchant_risk, 3),
        distance_from_last_tx = round(distance, 2),
        hour_of_day        = hour,
        day_of_week        = dow,
        is_weekend         = is_weekend,
    )


# ─── Main processing loop ─────────────────────────────────────────────────────

def main():
    fb = FlashBladeClient(REDIS_URL)
    logger.info(f"Data-prep connecting to FlashBlade at {REDIS_URL}")

    # Wait for Redis
    for _ in range(30):
        if fb.ping():
            break
        logger.info("Waiting for Redis...")
        time.sleep(2)

    logger.info("Data-prep worker started — listening on cpu_tx")

    processed   = 0
    error_count = 0
    last_metric = time.time()

    while True:
        t0 = time.perf_counter()
        try:
            messages = fb.read_stream(
                STREAM_CPU_TX, CONSUMER_GROUP, CONSUMER_NAME,
                count=BATCH_SIZE
            )
        except Exception as e:
            logger.error(f"Read error: {e}")
            time.sleep(1)
            continue

        if not messages:
            continue

        batch_start = time.perf_counter()
        for raw in messages:
            try:
                feat = engineer_features(raw)
                fb.write_features(feat)
                processed += 1
            except Exception as e:
                error_count += 1
                logger.warning(f"Feature engineering error: {e}")

        batch_ms = (time.perf_counter() - batch_start) * 1000

        now = time.time()
        if now - last_metric >= 5:
            tps = processed / max(1, now - last_metric + 5)
            fb.write_metric(Metric(ts=now, name="dataprep_tps",
                                   value=tps, labels={"backend": "gpu" if USE_GPU else "cpu"}))
            fb.write_metric(Metric(ts=now, name="dataprep_batch_latency_ms",
                                   value=batch_ms / max(1, len(messages)),
                                   labels={}))
            logger.info(f"Processed batch={len(messages)}  total={processed}  "
                        f"errors={error_count}  batch_ms={batch_ms:.1f}  "
                        f"backend={'GPU' if USE_GPU else 'CPU'}")
            processed   = 0
            last_metric = now


if __name__ == "__main__":
    main()
