"""
Synthetic multi-channel sensor stream generator.

Stands in for real ocean sensor telemetry (acoustic/sonar amplitude, temperature,
pressure, vibration) so the anomaly engine can be demoed and tested without a live
sensor feed. Swap this out for a real telemetry ingestion source in production —
the detector in detector.py doesn't care where the readings come from.
"""

import numpy as np

CHANNEL_PROFILES = {
    "acoustic": {"base": 40.0, "noise": 3.0, "unit": "dB"},
    "temperature": {"base": 4.2, "noise": 0.15, "unit": "\u00b0C"},
    "pressure": {"base": 300.0, "noise": 2.0, "unit": "bar"},
    "vibration": {"base": 0.02, "noise": 0.005, "unit": "g"},
}


def _make_point(channels, t, is_anomaly, rng):
    reading = {"t": t}
    for c in channels:
        profile = CHANNEL_PROFILES.get(c, {"base": 10.0, "noise": 1.0})
        value = profile["base"] + rng.normal(0, profile["noise"])
        if is_anomaly:
            magnitude = rng.uniform(4, 9) * profile["noise"]
            value += magnitude * rng.choice([-1, 1])
        reading[c] = round(float(value), 4)
    reading["_ground_truth_anomaly"] = bool(is_anomaly)
    return reading


def generate_stream(channels, n=200, anomaly_rate=0.05, seed=None):
    """Batch-generate n readings across the given channels."""
    rng = np.random.default_rng(seed)
    if anomaly_rate > 0:
        anomaly_indices = set(
            rng.choice(n, size=max(1, int(n * anomaly_rate)), replace=False)
        )
    else:
        anomaly_indices = set()
    return [_make_point(channels, t, t in anomaly_indices, rng) for t in range(n)]


def generate_single(channels, t, anomaly_rate, rng):
    """Generate one point, used for the live WebSocket stream."""
    is_anomaly = rng.random() < anomaly_rate
    return _make_point(channels, t, is_anomaly, rng)
