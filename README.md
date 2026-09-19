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
- **CORS open** — the Vercel frontend calls this from another origin.

## 🔑 Secrets

| Variable | Required | Notes |
| --- | --- | --- |
| `TOMTOM_API_KEY` | for `/api/traffic` | Set as a **Space secret**, never commit it |
| `SUPABASE_URL` | for `/api/traffic-history` | Project URL, e.g. `https://xxxx.supabase.co` |
| `SUPABASE_SECRET_KEY` | for `/api/traffic-history` | Service role / secret key — **never** put this in the frontend |

On Hugging Face: *Space → Settings → Variables and secrets → New secret*. Locally, drop it in a
gitignored `.env` — `python-dotenv` picks it up automatically.

Without the key, `/api/traffic` returns `{"error": "TomTom API key not found"}` and the frontend
shows a clear *feed offline* state instead of hanging.

## 🚀 Deploy

```bash
# GitHub (source of truth)
git push origin main

# Hugging Face Space (triggers a Docker rebuild)
git push space main
```

Add the Space remote once:

```bash
git remote add space https://huggingface.co/spaces/PRIYAREDDDY-CSE/hyderabad-urban-intelligence-api
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
