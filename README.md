# Deep Sensor Anomaly Engine

A generic anomaly-scoring engine for multi-channel sensor streams — acoustic/sonar
amplitude, temperature, pressure, vibration, or any other numeric telemetry channel.
It learns a baseline of "normal" from the first N readings, then scores every new
reading in real time using two combined signals:

1. **Rolling z-score** — fast, interpretable, catches sudden single-channel spikes
2. **Isolation Forest** — catches subtler multi-channel pattern shifts a single
   channel wouldn't flag on its own

Built with the same architecture as "Is This Real?": FastAPI backend + static
frontend, deployable to Railway + Vercel.

## Why this exists

Ocean sensor networks generate far more data than gets reviewed in real time — most
monitoring coverage is sparse, and by the time a human notices a problem in raw
telemetry, the event has already passed. This engine is a triage layer: instead of
showing an operator raw sensor noise, it surfaces a confidence score per reading so
attention goes to the events that actually matter.

## Project structure

```
deep-sensor-anomaly-engine/
├── backend/
│   ├── main.py           FastAPI app: /simulate, /analyze, /stream (WebSocket)
│   ├── detector.py        AnomalyEngine — the core scoring logic
│   ├── simulator.py       Synthetic sensor data generator (stand-in for real telemetry)
│   ├── requirements.txt
│   └── Procfile            Railway start command
└── frontend/
    └── index.html          Single-file dashboard (canvas waveform, live event log)
```

## Run locally

**Backend:**
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8420
```

**Frontend:** just open `frontend/index.html` in a browser, or serve it:
```bash
cd frontend
python3 -m http.server 8080
```
The frontend auto-detects localhost and points at `http://localhost:8420`.

## API

- `GET /health` — status check
- `GET /channels` — available synthetic channel types
- `POST /simulate?n=200&anomaly_rate=0.05&channels=acoustic,temperature,pressure,vibration`
  — generates a synthetic run and scores every point (batch demo mode)
- `POST /analyze` — bring your own data:
  ```json
  {
    "channels": ["acoustic", "temperature"],
    "readings": [{"t": 0, "acoustic": 40.1, "temperature": 4.2}, ...]
  }
  ```
- `WS /stream?channels=acoustic,temperature,pressure,vibration&anomaly_rate=0.06`
  — live one-reading-at-a-time stream, ~400ms interval

## Deploy (same pattern as your existing project)

**Backend → Railway:**
1. Push `backend/` to a GitHub repo (or a `backend` subfolder of one repo)
2. Railway → New Project → Deploy from GitHub → select the repo
3. Set root directory to `backend` if using a monorepo
4. Railway auto-detects `Procfile` and `requirements.txt` — no extra config needed
5. Copy the generated `*.up.railway.app` URL

**Frontend → Vercel:**
1. Push `frontend/` (or deploy just `index.html`) to Vercel as a static site
2. Before deploying, edit the `API_BASE` line near the top of the `<script>` tag in
   `index.html` and replace `https://YOUR-RAILWAY-APP.up.railway.app` with your real
   Railway URL
3. Deploy — Vercel will serve the static file directly

## Extending this into a real pitch demo

- Swap `simulator.py` for a real data source (CSV upload, live sensor feed, or a
  public ocean buoy dataset like NOAA's) to show the engine on real signals
- Add a "confidence threshold" slider so a reviewer can tune sensitivity live
- Log flagged events to a database instead of just the in-memory stream, so a
  history view persists across sessions

## Notes on the detection approach

`AnomalyEngine` is deliberately generic — it doesn't hardcode what "acoustic" or
"pressure" data should look like. It only needs a channel name and a baseline window
of readings to learn from, which is what makes it reusable across very different
sensor types without retraining a bespoke model per channel.
