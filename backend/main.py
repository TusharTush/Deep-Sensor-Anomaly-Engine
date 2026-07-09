import asyncio
from typing import Dict, List, Optional

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from detector import AnomalyEngine
from simulator import CHANNEL_PROFILES, generate_single, generate_stream

app = FastAPI(title="Deep Sensor Anomaly Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class AnalyzeRequest(BaseModel):
    channels: List[str]
    readings: List[Dict[str, float]]
    baseline_window: Optional[int] = 50


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/channels")
def channels():
    return {"available_channels": list(CHANNEL_PROFILES.keys())}


@app.post("/simulate")
def simulate(
    channels: str = "acoustic,temperature,pressure,vibration",
    n: int = 200,
    anomaly_rate: float = 0.05,
):
    """Batch demo mode: generate a synthetic run and score every point at once."""
    channel_list = channels.split(",")
    readings = generate_stream(channel_list, n=n, anomaly_rate=anomaly_rate)

    engine = AnomalyEngine(baseline_window=max(10, min(50, n // 4)))
    engine.fit_baseline(readings, channel_list)

    results = [{**r, **engine.score_point(r)} for r in readings]
    flagged_count = sum(1 for r in results if r["flagged"])

    return {
        "channels": channel_list,
        "total_points": n,
        "flagged_count": flagged_count,
        "readings": results,
    }


@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    """Bring-your-own-data mode: score a real uploaded reading set."""
    engine = AnomalyEngine(baseline_window=min(req.baseline_window, len(req.readings)))
    engine.fit_baseline(req.readings, req.channels)
    results = [{**r, **engine.score_point(r)} for r in req.readings]
    return {"readings": results}


@app.websocket("/stream")
async def stream(
    ws: WebSocket,
    channels: str = "acoustic,temperature,pressure,vibration",
    anomaly_rate: float = 0.06,
):
    """Live mode: streams one scored reading roughly every 400ms."""
    await ws.accept()
    channel_list = channels.split(",")
    rng = np.random.default_rng()

    baseline = generate_stream(channel_list, n=50, anomaly_rate=0.0, seed=1)
    engine = AnomalyEngine(baseline_window=50)
    engine.fit_baseline(baseline, channel_list)

    t = 50
    try:
        while True:
            point = generate_single(channel_list, t, anomaly_rate, rng)
            score = engine.score_point(point)
            await ws.send_json({**point, **score})
            t += 1
            await asyncio.sleep(0.4)
    except WebSocketDisconnect:
        pass
