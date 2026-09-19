---
title: Hyderabad Urban Intelligence API
emoji: 🚌
colorFrom: blue
colorTo: cyan
sdk: docker
app_port: 7860
pinned: false
---

# UrbanIQ backend (Hugging Face Space)

This Docker Space runs the Flask + YOLO API. The Vercel frontend should call:

`https://<your-username>-<space-name>.hf.space`

Check:

`https://<your-username>-<space-name>.hf.space/api/health`

## Create the Space

1. Push this repository (`HUP-Backend`) to GitHub.
2. Open [huggingface.co/new-space](https://huggingface.co/new-space).
3. Set **SDK** to **Docker**.
4. Connect this GitHub repo (`karthikganjam99kg/HUP-Backend`).
5. Hardware: **CPU basic** (free). Do not pick GPU unless you are paying.
6. After the build turns green, open `/api/health`. You want `"status": "ok"` and all three models `true`.
7. In Vercel, set `VITE_API_BASE_URL` to the Space URL (no trailing slash) and redeploy.

The first request after sleep can take 30–60 seconds while YOLO loads. Wake the Space with `/api/health` before a demo.
