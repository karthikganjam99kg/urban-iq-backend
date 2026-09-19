---
title: Hyderabad Urban Intelligence API
emoji: 🚌
colorFrom: blue
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

<div align="center">

# 🧠 URBAN**IQ** — NEURAL API

### The vision cortex behind SIH 26124

**Four YOLO models, one Flask brain, zero GPUs.**

[![Health](https://img.shields.io/badge/HEALTH-%2Fapi%2Fhealth-00e5ff?style=for-the-badge)](https://priyaredddy-cse-hyderabad-urban-intelligence-api.hf.space/api/health)
[![Docker Space](https://img.shields.io/badge/Hugging_Face-Docker_Space-ffcc4d?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/spaces/PRIYAREDDDY-CSE/hyderabad-urban-intelligence-api)
[![Flask](https://img.shields.io/badge/Flask-Gunicorn-000000?style=for-the-badge&logo=flask)](https://flask.palletsprojects.com)

</div>

---

## 🛰️ What runs here

This service is the heavy half of [UrbanIQ](https://github.com/karthikganjam99kg/urban-iq-frontend).
It takes a frame from a bus camera and returns civic intelligence: potholes with confidence,
garbage hotspots, vehicle counts, licence plates, rash-motion scores, and live congestion.

It runs on **CPU only**. Models load lazily on first request, so the container boots in seconds
and the first inference pays the warm-up cost.

## 🔌 Endpoints

| Method | Route | Returns |
| --- | --- | --- |
| `GET` | `/api/health` | Service status + which models are resident |
| `GET` | `/api/traffic` | Live TomTom flow: speed, free-flow speed, congestion score + level |
| `GET` | `/api/traffic-history` | Stored `traffic_realtime` rows from Supabase |
| `GET` | `/api/fleet` | Supabase bus positions, speeds, routes and telemetry freshness |
| `POST` | `/api/fleet/telemetry` | Ingest GPS readings for configured vehicles (token protected) |
| `GET` | `/api/demand-forecast` | Rolling one-hour demand proxy from Supabase `vehicle_density` history |
| `GET` | `/api/overview` | Aggregated service, fleet, model and alert status |
| `GET` | `/api/routes` | Route catalog, condition coverage and gated recommendation |
| `GET` | `/api/fitness` | Verified fitness routes/facilities with live traffic safety |
| `POST` | `/api/traffic-simulation` | Scenario calculated from recent traffic and vehicle-density data |
| `GET` | `/api/alerts` | Civic alerts from Supabase `alerts` |
| `GET` | `/api/incidents` | Incident records from Supabase `incidents` |
| `POST` | `/api/pothole-detect` | Boxed potholes with confidence (multipart upload) |
| `GET` | `/api/pothole-result` | Last annotated pothole image |
| `POST` | `/api/garbage-detect` | Waste objects + alert creation |
| `POST` | `/api/vehicle-detect` | Vehicle count, labels, confidence |
| `POST` | `/api/plate-detect` | Plate crops + OCR text (EasyOCR) |
| `POST` | `/api/rash-driving` | Motion scores across consecutive frames |

```bash
curl https://priyaredddy-cse-hyderabad-urban-intelligence-api.hf.space/api/health
# {"models":{"garbage":true,"pothole":true,"vehicle":true},"status":"ok"}
```

## 👁️ Model bay

| Model | Weight | Job |
| --- | --- | --- |
| Pothole | `weights/pothole2v.pt` | Road surface damage |
| Garbage | `src/garbage_ai/best.pt` | Waste / dumping detection |
| Vehicle | `yolo11n.pt` | Vehicle class + count |
| Plate | `weights/license_plate.pt` | Plate localisation, then EasyOCR |

## 🏗️ Design notes

- **Lazy loading** — each model is instantiated on first use, never at import time.
- **Writable caches** — on Spaces and serverless runtimes, `YOLO_CONFIG_DIR`, `HF_HOME`,
  `TORCH_HOME` and `MPLCONFIGDIR` are redirected to `/tmp`, because the app directory can be
  read-only.
- **Single worker, threaded** — one Gunicorn worker with threads keeps model memory to one copy.
- **Honest demand gating** — the rolling demand model only forecasts when it has at least six
  recent minute buckets spanning 30 minutes; otherwise it reports `collecting`.
- **Fleet freshness** — telemetry is `live` for five minutes, then `stale`; configured vehicles
  without any telemetry report `missing`.
- **Route recommendations** — a recommendation requires at least two routes with traffic,
  pothole, and waterlogging observations no older than 15 minutes.
- **Fitness ownership** — only active, verified `fitness_routes` and `sports_facilities` rows are
  returned. The traffic safety overlay is withheld when its latest snapshot is older than 10 minutes.
- **Simulation baseline** — scenarios require a traffic snapshot from the last 10 minutes, since
  the score is derived from the measured road speed. No fallback traffic score is used. The
  observed vehicle-density comparison is optional and reports its own `baseline.status` of `live`
  (within 10 minutes), `stale` (older, with `age_minutes`), or `missing`, so a quiet camera
  history never blocks a scenario or presents as a fresh count.
- **Detection ingestion** — vehicle, pothole, and garbage endpoints accept optional `bus_id` and
  `route_id` multipart fields. These links let detections accumulate route and demand evidence.
  Vehicle stills posted without a `bus_id` are stored as `OPERATOR-UPLOAD`, which keeps operator
  captures out of per-vehicle fleet views while still recording a real observation.
- **CORS open** — the Vercel frontend calls this from another origin.

## 🔑 Secrets

| Variable | Required | Notes |
| --- | --- | --- |
| `TOMTOM_API_KEY` | for `/api/traffic` | Set as a **Space secret**, never commit it |
| `SUPABASE_URL` | for `/api/traffic-history` | Project URL, e.g. `https://xxxx.supabase.co` |
| `SUPABASE_SECRET_KEY` | for traffic history, alerts, incidents | Service role / secret key — **never** put this in the frontend |
| `FLEET_INGEST_TOKEN` | for `/api/fleet/telemetry` | Shared secret for GPS devices. Ingestion returns 503 while unset, so the endpoint is never open |

On Hugging Face: *Space → Settings → Variables and secrets → New secret*. Locally, drop it in a
gitignored `.env` — `python-dotenv` picks it up automatically.

## 🛰️ Sending fleet telemetry

`/api/fleet` reports `live` only when a vehicle reported within the last five minutes. Positions
have to be pushed in; nothing generates them. Post one reading or a batch of up to 200:

```bash
curl -X POST https://<space>.hf.space/api/fleet/telemetry \
  -H "X-Ingest-Token: $FLEET_INGEST_TOKEN" \
  -H "Content-Type: application/json" \
  -d '[{"bus_id":"HYD-BUS-001","latitude":17.3850,"longitude":78.4867,"speed":24,"heading":90,"status":"ACTIVE"}]'
```

- `bus_id`, `latitude` and `longitude` are required; `bus_id` must exist in `fleet_vehicles`.
- `route_id` is taken from the vehicle catalog when omitted.
- `speed`, `heading` and `status` are optional. Status is stored only when reported, so the UI
  shows `UNKNOWN` rather than assuming a bus is in service.
- Readings are appended, so `/api/fleet` always reads the newest row per vehicle.

Without the key, `/api/traffic` returns `{"error": "TomTom API key not found"}` and the frontend
shows a clear *feed offline* state instead of hanging.

## 🚀 Deploy

Push to `main` on GitHub. A GitHub Action (`sync-to-space.yml`) mirrors the tree to
the Hugging Face Space and Docker rebuilds from that upload.

```bash
git push origin main
```

One-time setup: create a Hugging Face **write** token that can update
`PRIYAREDDDY-CSE/hyderabad-urban-intelligence-api`, then add it as the GitHub
secret `HF_TOKEN` on this repo. The token must belong to an account that already
has write access on that Space (Priya, or a collaborator she added).

The Space uses the Dockerfile, not a Procfile. Manual Space push still works if you need it:

```bash
git remote add space https://huggingface.co/spaces/PRIYAREDDDY-CSE/hyderabad-urban-intelligence-api
git push space main
```

Hardware: **CPU basic**. A GPU is not required for these model sizes.

## 🧪 Local run

```bash
pip install -r requirements.txt
echo "TOMTOM_API_KEY=your_key" > .env
gunicorn --bind 127.0.0.1:5001 --workers 1 --threads 2 --timeout 300 src.backend.app:app
```

Port 5001, not 5000 — macOS AirPlay Receiver claims 5000 and returns 403.

<div align="center">

**Smart India Hackathon · Problem Statement 26124**

</div>
