"""
modules/inference/main.py
Inference module — reads from cpu_features (CPU pipeline) and gpu_tx (GPU pipeline).
Calls NVIDIA Triton Inference Server for GNN + XGBoost fraud scoring.
Falls back to local XGBoost model if Triton is unavailable (dev mode).
Writes FraudScore objects to cpu_scores / gpu_scores streams.
"""
import os
import sys
import time
import json
import logging
import threading
import numpy as np
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from shared.featurebus.client import (
    FeatureBusClient, FraudScore, Metric,
    STREAM_CPU_FEATURES, STREAM_GPU_TX,
    STREAM_CPU_SCORES, STREAM_GPU_SCORES
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [inference] %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ─── Config ───────────────────────────────────────────────────────────────────
REDIS_URL       = os.getenv("REDIS_URL", "redis://redis:6379")
TRITON_URL      = os.getenv("TRITON_URL", "triton:8001")
TRITON_MODEL    = os.getenv("TRITON_MODEL", "fraud_ensemble")
FRAUD_THRESHOLD = float(os.getenv("FRAUD_THRESHOLD", "0.7"))
BATCH_SIZE      = int(os.getenv("INFERENCE_BATCH_SIZE", "64"))
GPU_ENABLED     = os.getenv("GPU_ENABLED", "true").lower() == "true"

# ─── Feature columns (must match training) ───────────────────────────────────
CPU_FEATURE_COLS = [
    "amount_log", "time_since_last_tx", "tx_velocity_1h",
    "amount_velocity_1h", "merchant_risk_score", "distance_from_last_tx",
    "hour_of_day", "day_of_week", "is_weekend", "is_online", "amount"
]

GPU_FEATURE_COLS = [
    "amount", "prev_tx_amount", "is_online",
    "merchant_lat", "merchant_lon"
]

SHAPLEY_FEATURE_NAMES = CPU_FEATURE_COLS  # for explainability


# ─── Triton client ────────────────────────────────────────────────────────────

class TritonScoringClient:
    def __init__(self, url: str, model_name: str):
        self.url        = url
        self.model_name = model_name
        self.client     = None
        self._connect()

    def _connect(self):
        try:
            import tritonclient.grpc as triton_grpc
            self.client = triton_grpc.InferenceServerClient(url=self.url, verbose=False)
            if self.client.is_server_ready():
                logger.info(f"✅ Triton connected at {self.url}")
            else:
                logger.warning("Triton server not ready — will retry")
                self.client = None
        except ImportError:
            logger.warning("tritonclient not installed — using fallback scorer")
        except Exception as e:
            logger.warning(f"Triton connection failed: {e} — using fallback scorer")

    def score_batch(self, features: np.ndarray) -> np.ndarray:
        """Returns array of fraud scores [0-1]."""
        if self.client is None:
            self._connect()
        if self.client is None:
            return None

        try:
            import tritonclient.grpc as triton_grpc
            input0 = triton_grpc.InferInput("input__0", features.shape, "FP32")
            input0.set_data_from_numpy(features.astype(np.float32))
            output0 = triton_grpc.InferRequestedOutput("output__0")
            result  = self.client.infer(
                model_name=self.model_name,
                inputs=[input0],
                outputs=[output0]
            )
            return result.as_numpy("output__0").flatten()
        except Exception as e:
            logger.warning(f"Triton inference error: {e}")
            return None


# ─── Fallback local scorer ────────────────────────────────────────────────────

class FallbackScorer:
    """
    Lightweight heuristic scorer for dev/testing without Triton.
    Mimics XGBoost output distribution using hand-crafted rules based on
    the feature engineering from the NVIDIA NIM blueprint.
    """
    def score_batch(self, features: np.ndarray, col_names: list) -> np.ndarray:
        scores = np.zeros(len(features), dtype=np.float32)
        col_map = {c: i for i, c in enumerate(col_names)}

        for i, row in enumerate(features):
            score = 0.05  # base rate

            # Amount log (large amounts → higher risk)
            if "amount_log" in col_map:
                al = row[col_map["amount_log"]]
                if al > 6.0:    score += 0.25
                elif al > 5.0:  score += 0.10

            # Merchant risk
            if "merchant_risk_score" in col_map:
                score += row[col_map["merchant_risk_score"]] * 0.30

            # Velocity
            if "tx_velocity_1h" in col_map:
                vel = row[col_map["tx_velocity_1h"]]
                if vel > 10:    score += 0.20
                elif vel > 5:   score += 0.10

            # Online + high amount
            if "is_online" in col_map and "amount" in col_map:
                if row[col_map["is_online"]] and row[col_map["amount"]] > 500:
                    score += 0.15

            # Time of day (overnight)
            if "hour_of_day" in col_map:
                h = int(row[col_map["hour_of_day"]])
                if h in (0, 1, 2, 3):  score += 0.10

            scores[i] = min(score + np.random.normal(0, 0.03), 1.0)

        return np.clip(scores, 0, 1)


# ─── Inference worker ─────────────────────────────────────────────────────────

class InferenceWorker:
    def __init__(self, pipeline: str, stream_in: str, stream_out: str,
                 feature_cols: list, fb: FeatureBusClient,
                 triton: TritonScoringClient, fallback: FallbackScorer):
        self.pipeline     = pipeline
        self.stream_in    = stream_in
        self.stream_out   = stream_out
        self.feature_cols = feature_cols
        self.fb           = fb
        self.triton       = triton
        self.fallback     = fallback
        self.group        = f"inference-{pipeline}-group"
        self.consumer     = f"inference-{pipeline}-worker"

        self._scored      = 0
        self._fraud_count = 0
        self._fraud_value = 0.0
        self._last_metric = time.time()

    def _extract_features(self, raw: dict) -> Optional[tuple[np.ndarray, dict]]:
        try:
            row = []
            for col in self.feature_cols:
                val = raw.get(col, 0)
                row.append(float(val))
            return np.array(row, dtype=np.float32), raw
        except Exception as e:
            logger.warning(f"Feature extraction error: {e}")
            return None

    def _score_batch(self, feature_matrix: np.ndarray) -> np.ndarray:
        t0     = time.perf_counter()
        scores = self.triton.score_batch(feature_matrix) if self.triton else None
        if scores is None:
            scores = self.fallback.score_batch(feature_matrix, self.feature_cols)
        latency_ms = (time.perf_counter() - t0) * 1000
        return scores, latency_ms

    def _top_shapley(self, feature_row: np.ndarray) -> tuple[str, float]:
        """Approximate Shapley: return feature with max absolute deviation from mean."""
        means = np.array([0.5] * len(self.feature_cols))  # simplified baseline
        diffs = np.abs(feature_row - means)
        idx   = int(np.argmax(diffs))
        return self.feature_cols[idx], float(diffs[idx])

    def run_once(self):
        messages = self.fb.read_stream(
            self.stream_in, self.group, self.consumer, count=BATCH_SIZE
        )
        if not messages:
            return

        rows, metas = [], []
        for raw in messages:
            result = self._extract_features(raw)
            if result:
                feat, meta = result
                rows.append(feat)
                metas.append(meta)

        if not rows:
            return

        matrix = np.stack(rows)
        scores, latency_ms = self._score_batch(matrix)

        now = time.time()
        for feat_row, meta, score in zip(rows, metas, scores):
            is_fraud = bool(score >= FRAUD_THRESHOLD)
            amount   = float(meta.get("amount", 0))
            top_feat, top_val = self._top_shapley(feat_row)

            fraud_score = FraudScore(
                tx_id              = meta.get("tx_id", ""),
                timestamp          = now,
                pipeline           = self.pipeline,
                fraud_score        = round(float(score), 4),
                is_fraud           = is_fraud,
                amount             = amount,
                shapley_top_feature = top_feat,
                shapley_top_value  = round(top_val, 4),
                inference_latency_ms = round(latency_ms / len(rows), 3),
            )
            self.fb.write_score(fraud_score)

            self._scored += 1
            if is_fraud:
                self._fraud_count += 1
                self._fraud_value += amount

        # Emit metrics every 5s
        if now - self._last_metric >= 5:
            tps = self._scored / 5.0
            self.fb.write_metric(Metric(ts=now, name="inference_tps",
                                        value=tps, labels={"pipeline": self.pipeline}))
            self.fb.write_metric(Metric(ts=now, name="inference_fraud_count",
                                        value=float(self._fraud_count),
                                        labels={"pipeline": self.pipeline}))
            self.fb.write_metric(Metric(ts=now, name="inference_fraud_value_usd",
                                        value=round(self._fraud_value, 2),
                                        labels={"pipeline": self.pipeline}))
            self.fb.write_metric(Metric(ts=now, name="inference_latency_ms",
                                        value=latency_ms / max(1, len(rows)),
                                        labels={"pipeline": self.pipeline}))
            logger.info(f"[{self.pipeline.upper()}] scored={self._scored}  "
                        f"fraud={self._fraud_count}  "
                        f"fraud_value=${self._fraud_value:,.2f}  "
                        f"latency={latency_ms/len(rows):.2f}ms")
            self._scored      = 0
            self._fraud_count = 0
            self._fraud_value = 0.0
            self._last_metric = now


def main():
    fb       = FeatureBusClient(REDIS_URL)
    triton   = TritonScoringClient(TRITON_URL, TRITON_MODEL) if GPU_ENABLED else None
    fallback = FallbackScorer()

    # Wait for Redis
    for _ in range(30):
        if fb.ping():
            break
        logger.info("Waiting for Redis...")
        time.sleep(2)

    cpu_worker = InferenceWorker(
        "cpu", STREAM_CPU_FEATURES, STREAM_CPU_SCORES,
        CPU_FEATURE_COLS, fb, triton, fallback
    )
    gpu_worker = InferenceWorker(
        "gpu", STREAM_GPU_TX, STREAM_GPU_SCORES,
        GPU_FEATURE_COLS, fb, triton, fallback
    )

    logger.info("Inference workers started")

    def run_worker(worker):
        while True:
            try:
                worker.run_once()
            except Exception as e:
                logger.error(f"Worker error ({worker.pipeline}): {e}")
                time.sleep(1)

    cpu_thread = threading.Thread(target=run_worker, args=(cpu_worker,), daemon=True)
    gpu_thread = threading.Thread(target=run_worker, args=(gpu_worker,), daemon=True)

    cpu_thread.start()
    gpu_thread.start()

    cpu_thread.join()
    gpu_thread.join()


if __name__ == "__main__":
    main()
