"""
modules/generator/main.py
Synthetic transaction generator.
Loads Kaggle credit-card fraud CSV as a seed for realistic statistical profiles,
then uses Faker to synthesise realistic card/merchant/cardholder data.
Writes to BOTH cpu_tx and gpu_tx streams on the FlashBlade.
Responds to Redis pub/sub control messages for STRESS mode.
"""
import os
import sys
import time
import uuid
import math
import json
import logging
import threading
from pathlib import Path

import numpy as np
import pandas as pd
from faker import Faker

# Make shared importable when running directly
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from shared.featurebus.client import FlashBladeClient, RawTransaction, Metric

logging.basicConfig(level=logging.INFO, format="%(asctime)s [generator] %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ─── Config ───────────────────────────────────────────────────────────────────
REDIS_URL          = os.getenv("REDIS_URL", "redis://redis:6379")
KAGGLE_PATH        = os.getenv("KAGGLE_DATASET_PATH", "/data/kaggle_seed.csv")
TX_RATE_NORMAL     = float(os.getenv("TX_RATE_NORMAL", "100"))    # tx/sec
TX_RATE_STRESS     = float(os.getenv("TX_RATE_STRESS", "2000"))   # tx/sec
GPU_ENABLED        = os.getenv("GPU_ENABLED", "true").lower() == "true"
GPU_TX_FRACTION    = float(os.getenv("GPU_TX_FRACTION", "0.5"))   # fraction sent to gpu pipeline

fake = Faker()
Faker.seed(42)
rng = np.random.default_rng(42)

# ─── Seed data ─────────────────────────────────────────────────────────────

MERCHANT_CATEGORIES = [
    "grocery_pos", "gas_transport", "home", "shopping_net",
    "entertainment", "food_dining", "travel", "health_fitness",
    "personal_care", "kids_pets", "misc_net", "misc_pos"
]

MERCHANT_RISK = {cat: rng.random() * 0.4 for cat in MERCHANT_CATEGORIES}
MERCHANT_RISK["shopping_net"] = 0.65
MERCHANT_RISK["misc_net"] = 0.72
MERCHANT_RISK["travel"] = 0.55


def load_kaggle_profiles(path: str) -> dict:
    """Extract statistical profiles from Kaggle dataset for realistic amount distributions."""
    try:
        df = pd.read_csv(path)
        # Kaggle CC fraud dataset columns: Time, V1-V28, Amount, Class
        profiles = {
            "amount_mean": float(df["Amount"].mean()),
            "amount_std":  float(df["Amount"].std()),
            "amount_p95":  float(df["Amount"].quantile(0.95)),
            "fraud_rate":  float(df["Class"].mean()),
            "n_samples":   len(df),
        }
        # Build empirical amount distribution
        profiles["amount_samples"] = df["Amount"].clip(upper=5000).tolist()
        logger.info(f"Loaded Kaggle seed: {profiles['n_samples']} rows, "
                    f"fraud_rate={profiles['fraud_rate']:.3%}")
        return profiles
    except FileNotFoundError:
        logger.warning(f"Kaggle seed not found at {path}, using synthetic defaults")
        return {
            "amount_mean": 88.35, "amount_std": 250.12,
            "amount_p95": 350.0, "fraud_rate": 0.00172,
            "amount_samples": list(rng.exponential(scale=88, size=10_000).clip(0.5, 5000))
        }


def generate_transaction(profiles: dict, cardholder_pool: list,
                          merchant_pool: list, pipeline: str) -> RawTransaction:
    """Synthesise a single realistic transaction."""
    cardholder = rng.choice(cardholder_pool)
    merchant   = rng.choice(merchant_pool)
    now        = time.time()

    # Sample amount from empirical distribution
    amount = float(rng.choice(profiles["amount_samples"]))
    # Occasionally inject a fraud-like spike
    if rng.random() < profiles["fraud_rate"] * 10:
        amount *= rng.uniform(3, 15)

    prev_amount = float(rng.choice(profiles["amount_samples"]))
    prev_ts     = now - rng.uniform(60, 86_400)

    return RawTransaction(
        tx_id             = str(uuid.uuid4()),
        timestamp         = now,
        amount            = round(amount, 2),
        card_number       = cardholder["last4"],
        merchant_id       = merchant["id"],
        merchant_category = merchant["category"],
        merchant_lat      = merchant["lat"],
        merchant_lon      = merchant["lon"],
        cardholder_id     = cardholder["id"],
        prev_tx_amount    = round(prev_amount, 2),
        prev_tx_timestamp = prev_ts,
        is_online         = bool(rng.random() > 0.6),
        currency          = "USD",
        pipeline          = pipeline,
    )


def build_pools(n_cardholders=5000, n_merchants=500):
    """Build stable fake cardholder and merchant pools."""
    logger.info("Building cardholder/merchant pools...")
    cardholders = [
        {"id": str(uuid.uuid4()),
         "last4": fake.credit_card_number()[-4:],
         "name": fake.name(),
         "lat":  float(rng.uniform(25, 48)),
         "lon":  float(rng.uniform(-122, -70))}
        for _ in range(n_cardholders)
    ]
    merchants = [
        {"id": str(uuid.uuid4()),
         "name": fake.company(),
         "category": rng.choice(MERCHANT_CATEGORIES),
         "lat": float(rng.uniform(25, 48)),
         "lon": float(rng.uniform(-122, -70))}
        for _ in range(n_merchants)
    ]
    return cardholders, merchants


# ─── Generator loop ──────────────────────────────────────────────────────────

class TxGenerator:
    def __init__(self):
        self.fb     = FlashBladeClient(REDIS_URL)
        self.rate   = TX_RATE_NORMAL        # tx/sec (mutable via stress control)
        self.stress = False
        self._stop  = threading.Event()

        profiles             = load_kaggle_profiles(KAGGLE_PATH)
        self.profiles        = profiles
        self.cardholders, self.merchants = build_pools()

        # Metrics accumulators
        self._tx_count   = 0
        self._last_metric_ts = time.time()

        logger.info(f"Generator initialised — normal_rate={TX_RATE_NORMAL}/s  "
                    f"stress_rate={TX_RATE_STRESS}/s  GPU_ENABLED={GPU_ENABLED}")

    def _listen_control(self):
        """Background thread: listen for stress/config signals."""
        ps = self.fb.subscribe_control()
        for msg in ps.listen():
            if self._stop.is_set():
                break
            try:
                data = json.loads(msg["data"])
                if data.get("action") == "stress_enable":
                    self.rate   = TX_RATE_STRESS
                    self.stress = True
                    logger.info("🔴 STRESS MODE ON")
                elif data.get("action") == "stress_disable":
                    self.rate   = TX_RATE_NORMAL
                    self.stress = False
                    logger.info("🟢 STRESS MODE OFF")
                elif data.get("action") == "set_rate":
                    self.rate = float(data["rate"])
                    logger.info(f"Rate set to {self.rate}/s")
            except Exception as e:
                logger.warning(f"Control msg error: {e}")

    def _emit_metrics(self):
        now = time.time()
        elapsed = now - self._last_metric_ts
        if elapsed >= 5:
            tps = self._tx_count / elapsed
            self.fb.write_metric(Metric(ts=now, name="generator_tps", value=tps,
                                        labels={"stress": str(self.stress)}))
            self.fb.write_metric(Metric(ts=now, name="generator_tx_total",
                                        value=float(self._tx_count),
                                        labels={}))
            logger.info(f"TPS={tps:.1f}  total_tx={self._tx_count}  stress={self.stress}")
            self._tx_count = 0
            self._last_metric_ts = now

    def run(self):
        ctrl_thread = threading.Thread(target=self._listen_control, daemon=True)
        ctrl_thread.start()

        logger.info("Starting transaction generation loop...")
        while not self._stop.is_set():
            t0 = time.perf_counter()

            # Decide pipeline routing
            use_gpu = GPU_ENABLED and (rng.random() < GPU_TX_FRACTION)
            pipeline = "gpu" if use_gpu else "cpu"

            tx = generate_transaction(self.profiles, self.cardholders,
                                      self.merchants, pipeline)
            try:
                self.fb.write_raw_tx(tx)
                self._tx_count += 1
            except Exception as e:
                logger.error(f"FB write error: {e}")

            self._emit_metrics()

            # Rate limiting
            elapsed = time.perf_counter() - t0
            sleep_t = max(0, (1.0 / self.rate) - elapsed)
            if sleep_t > 0:
                time.sleep(sleep_t)

    def stop(self):
        self._stop.set()


if __name__ == "__main__":
    gen = TxGenerator()
    try:
        gen.run()
    except KeyboardInterrupt:
        gen.stop()
        logger.info("Generator stopped.")
