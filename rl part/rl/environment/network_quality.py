import numpy as np
import pandas as pd
from rl.utils.logger import setup_logger

logger = setup_logger("network_quality")


class NetworkQualityCalculator:
    """
    Calculates a consolidated, normalized network quality score [0.0, 1.0]
    dynamically based on available cell metrics.

    Priority order for score calculation:
      1. Signal metrics (RSSI, RSRP, SINR) if available
      2. Throughput if available
      3. Latency / packet_loss if available
      4. Fallback: network_data_confidence if all above are NaN
      5. Fallback: nearest_tower_distance-based proximity score

    This ensures network_score always has meaningful variation across cells
    even when the OpenCelliD/OpenTopography API keys are not yet configured
    (the pipeline still writes nearest_tower_distance and
    network_data_confidence from the ITU-R estimator).
    """

    def __init__(self, config: dict = None):
        # Component weights (renormalized if metrics are missing)
        self.weights = {
            "signal":      0.40,
            "latency":     0.20,
            "packet_loss": 0.30,
            "throughput":  0.10,
        }

        # Min/max thresholds for each raw metric
        self.thresholds = {
            "rssi":        {"min": -120.0, "max": -30.0},   # dBm
            "rsrp":        {"min": -140.0, "max": -60.0},   # dBm
            "sinr":        {"min": -10.0,  "max": 30.0},    # dB
            "latency":     {"min": 10.0,   "max": 500.0},   # ms (lower is better)
            "packet_loss": {"min": 0.0,    "max": 0.5},     # ratio (lower is better)
            "throughput":  {"min": 0.1,    "max": 100.0},   # Mbps
            # Fallback metrics
            "nearest_tower_distance": {"min": 50.0, "max": 5000.0},  # meters
        }

        self.outage_threshold = 0.3
        if config:
            net_cfg               = config.get("network", {})
            self.outage_threshold = float(net_cfg.get("outage_threshold", 0.3))

    # ─────────────────────────────────────────────────────────────────────────
    # Individual metric scorers
    # ─────────────────────────────────────────────────────────────────────────

    def _score(self, key: str, val: float, invert: bool = False) -> float:
        """Linear normalization, optionally inverted for 'lower is better' metrics."""
        t = self.thresholds[key]
        s = (val - t["min"]) / (t["max"] - t["min"])
        s = float(np.clip(s, 0.0, 1.0))
        return 1.0 - s if invert else s

    def compute_rssi_score(self, val: float) -> float:
        return self._score("rssi", val)

    def compute_rsrp_score(self, val: float) -> float:
        return self._score("rsrp", val)

    def compute_sinr_score(self, val: float) -> float:
        return self._score("sinr", val)

    def compute_latency_score(self, val: float) -> float:
        return self._score("latency", val, invert=True)

    def compute_packet_loss_score(self, val: float) -> float:
        return self._score("packet_loss", val, invert=True)

    def compute_throughput_score(self, val: float) -> float:
        return self._score("throughput", val)

    def compute_tower_proximity_score(self, dist_m: float) -> float:
        """Closer tower → higher score (1.0 at 50 m, 0.0 at 5000 m)."""
        return self._score("nearest_tower_distance", dist_m, invert=True)

    # ─────────────────────────────────────────────────────────────────────────
    # Main entry point
    # ─────────────────────────────────────────────────────────────────────────

    def calculate(self, cell_data: dict) -> dict:
        """
        Calculates the normalized network quality score and reports metadata.

        Args:
            cell_data: Dictionary of raw cell features (RSSI, latency, …)

        Returns:
            {
                "network_score":      float [0.0, 1.0],
                "confidence":         float [0.0, 1.0],
                "available_metrics":  list[str],
            }
        """
        available_metrics: list[str] = []
        scores: dict[str, float]     = {}

        # ── 1. Signal component (RSSI, RSRP, SINR) ───────────────────────────
        signal_scores: list[float] = []

        rssi = cell_data.get("rssi")
        if rssi is not None and not pd.isnull(rssi):
            signal_scores.append(self.compute_rssi_score(float(rssi)))
            available_metrics.append("rssi")

        rsrp = cell_data.get("rsrp")
        if rsrp is not None and not pd.isnull(rsrp):
            signal_scores.append(self.compute_rsrp_score(float(rsrp)))
            available_metrics.append("rsrp")

        sinr = cell_data.get("sinr")
        if sinr is not None and not pd.isnull(sinr):
            signal_scores.append(self.compute_sinr_score(float(sinr)))
            available_metrics.append("sinr")

        if signal_scores:
            scores["signal"] = float(np.mean(signal_scores))

        # ── 2. Latency component ─────────────────────────────────────────────
        latency = cell_data.get("latency")
        if latency is not None and not pd.isnull(latency):
            scores["latency"] = self.compute_latency_score(float(latency))
            available_metrics.append("latency")

        # ── 3. Packet loss component ──────────────────────────────────────────
        packet_loss = cell_data.get("packet_loss")
        if packet_loss is not None and not pd.isnull(packet_loss):
            scores["packet_loss"] = self.compute_packet_loss_score(float(packet_loss))
            available_metrics.append("packet_loss")

        # ── 4. Throughput component ──────────────────────────────────────────
        throughput = cell_data.get("throughput")
        if throughput is not None and not pd.isnull(throughput):
            scores["throughput"] = self.compute_throughput_score(float(throughput))
            available_metrics.append("throughput")

        # ── 5. Compute weighted score from available metrics ──────────────────
        active_weight_total = sum(
            self.weights[k] for k in self.weights if k in scores
        )

        if active_weight_total > 0.0:
            normalized_score = sum(
                (self.weights[k] / active_weight_total) * scores[k]
                for k in scores if k in self.weights
            )
        else:
            # ── 6. No primary metrics — use fallback signals ─────────────────
            #
            # Fallback priority:
            #   a) network_data_confidence  (set by the ITU-R estimator)
            #   b) nearest_tower_distance   (always available if towers were downloaded)
            #   c) 0.0  (truly unknown — no tower data at all)
            #
            fallback_score: float | None = None

            conf = cell_data.get("network_data_confidence")
            if conf is not None and not pd.isnull(conf):
                # Confidence already in [0, 1]
                fallback_score = float(np.clip(conf, 0.0, 1.0))
                available_metrics.append("network_data_confidence")

            if fallback_score is None:
                tower_dist = cell_data.get("nearest_tower_distance")
                if tower_dist is not None and not pd.isnull(tower_dist):
                    fallback_score = self.compute_tower_proximity_score(float(tower_dist))
                    available_metrics.append("nearest_tower_distance")

            normalized_score = fallback_score if fallback_score is not None else 0.0

        # ── 7. Confidence ────────────────────────────────────────────────────
        confidence = cell_data.get("network_data_confidence")
        if confidence is None or pd.isnull(confidence):
            # If we have some primary metrics, report moderate confidence
            confidence = 0.5 if available_metrics else 0.0
        else:
            confidence = float(np.clip(confidence, 0.0, 1.0))

        return {
            "network_score":     float(np.clip(normalized_score, 0.0, 1.0)),
            "confidence":        float(confidence),
            "available_metrics": available_metrics,
        }
