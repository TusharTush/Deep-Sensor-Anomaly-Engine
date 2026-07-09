"""
Generic multi-channel anomaly-scoring engine.

Designed to work on any numeric sensor stream — acoustic amplitude, pressure,
temperature, vibration, or any other channel a sensor network reports — without
being told in advance what "normal" looks like for that deployment. It learns a
baseline from the first N readings, then scores every subsequent point.

Two independent signals are combined so a single noisy channel can't dominate:
  1. Rolling z-score     -> fast, interpretable, catches sudden single-channel spikes
  2. Isolation Forest     -> catches subtler multi-channel pattern shifts that no
                             single channel would flag on its own
"""

import numpy as np
from sklearn.ensemble import IsolationForest


class AnomalyEngine:
    def __init__(self, baseline_window: int = 50, contamination: float = 0.05):
        self.baseline_window = baseline_window
        self.contamination = contamination
        self.channel_stats = {}
        self.iso_forest = None
        self.channels = []
        self.if_min = 0.0
        self.if_max = 1.0

    def fit_baseline(self, readings, channels):
        """Learn what 'normal' looks like from the first baseline_window readings."""
        self.channels = channels
        window = readings[: self.baseline_window]
        X = np.array([[r[c] for c in channels] for r in window])

        for i, c in enumerate(channels):
            self.channel_stats[c] = (float(X[:, i].mean()), float(X[:, i].std()) + 1e-6)

        self.iso_forest = IsolationForest(
            n_estimators=100, contamination=self.contamination, random_state=42
        )
        self.iso_forest.fit(X)

        scores = self.iso_forest.score_samples(X)
        self.if_min, self.if_max = float(scores.min()), float(scores.max())

    def score_point(self, reading):
        """Score a single reading against the learned baseline. Returns confidence 0-1."""
        vec = np.array([[reading[c] for c in self.channels]])

        z_scores = {}
        for c in self.channels:
            mean, std = self.channel_stats[c]
            z_scores[c] = abs((reading[c] - mean) / std)
        z_component = min(max(z_scores.values()) / 5.0, 1.0)

        raw_score = self.iso_forest.score_samples(vec)[0]  # higher = more "normal"
        span = (self.if_max - self.if_min) + 1e-9
        if_component = float(np.clip(1 - (raw_score - self.if_min) / span, 0, 1))

        confidence = round(0.5 * z_component + 0.5 * if_component, 4)
        return {
            "confidence": confidence,
            "flagged": confidence > 0.55,
            "channel_breakdown": {c: round(v, 3) for c, v in z_scores.items()},
        }
