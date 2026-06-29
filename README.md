# 🔥 FireWatch — Real-Time Forest Fire Detection System

A production-grade wildfire detection dashboard combining **YOLOv8 computer vision** on live camera feeds with **NASA FIRMS satellite hotspot data** on an interactive map.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green?style=flat-square&logo=fastapi)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-orange?style=flat-square)
![NASA FIRMS](https://img.shields.io/badge/NASA-FIRMS-red?style=flat-square)

---

## ✨ Features

| Feature | Description |
|---|---|
| 🎥 **Live Detection** | YOLOv8 fire detection on RTSP streams & uploaded videos |
| 🛰 **Satellite Data** | NASA FIRMS MODIS/VIIRS hotspots refreshed every 15 min |
| 🗺 **Unified Map** | Leaflet map showing satellite hotspots + camera locations |
| 📊 **Analytics** | Detection timeline, severity charts, 7-day history, per-source breakdown |
| ⚠️ **Alert System** | Real-time WebSocket alerts with severity levels (Watch/Warning/Emergency) |
| 💾 **Persistence** | Alerts stored in SQLite — survive restarts |
| 📥 **CSV Export** | Download full alert history as a timestamped CSV |
| 🔔 **Webhooks** | Optional Slack/Teams/custom webhook for emergency notifications |

---

## 🏗 Architecture

```
┌─────────────────────────────────────────────┐
│                  Browser                     │
│  Live Feeds │ Satellite Map │ Analytics      │
└──────────────────┬──────────────────────────┘
                   │ WebSocket + REST
┌──────────────────▼──────────────────────────┐
│            FastAPI Backend                   │
│  ┌──────────┐ ┌──────────┐ ┌────────────┐  │
│  │ Stream   │ │ YOLOv8   │ │ FIRMS      │  │
│  │ Manager  │→│ Detector │ │ Service    │  │
│  └──────────┘ └──────────┘ └────────────┘  │
│  ┌──────────┐ ┌──────────────────────────┐  │
│  │ Alert    │ │ SQLite DB                │  │
│  │ Service  │→│ (fire_detection.db)      │  │
│  └──────────┘ └──────────────────────────┘  │
└─────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### 1. Clone & set up environment

```bash
git clone https://github.com/YOUR_USERNAME/forest-fire-detection.git
cd forest-fire-detection

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r backend/requirements.txt
```

### 2. Configure secrets

```bash
cp .env.example .env
# Edit .env and fill in your keys
```

Get your free NASA FIRMS key at: https://firms.modaps.eosdis.nasa.gov/api/map_key/

### 3. Add a fire sample video (optional demo)

```bash
# Place any fire video in data/videos/
cp your_fire_video.mp4 data/videos/
```

### 4. Run

```bash
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

Open **http://localhost:8000**

---

## 🐳 Docker

```bash
docker-compose up --build
```

---

## 📁 Project Structure

```
forest-fire-detection/
├── backend/
│   ├── api/
│   │   └── routes.py          # FastAPI endpoints & WebSocket
│   ├── core/
│   │   ├── detector.py        # YOLOv8 fire detection
│   │   └── stream_manager.py  # RTSP/file stream handling
│   ├── services/
│   │   ├── alert_service.py   # Alert creation & SQLite persistence
│   │   └── firms_service.py   # NASA FIRMS API integration
│   └── main.py
├── frontend/
│   └── index.html             # Single-page dashboard (vanilla JS)
├── data/
│   ├── videos/                # Drop fire videos here for auto-loading
│   └── outputs/               # Annotated frame outputs
├── .env.example               # Config template (copy to .env)
├── docker-compose.yml
└── yolov8n.pt                 # YOLOv8 model weights
```

---

## ⚙️ Configuration

| Variable | Default | Description |
|---|---|---|
| `FIRMS_MAP_KEY` | — | NASA FIRMS API key (required for satellite data) |
| `FIRMS_SOURCE` | `MODIS_NRT` | Satellite source: `MODIS_NRT`, `VIIRS_SNPP_NRT`, `VIIRS_NOAA20_NRT` |
| `FIRMS_DAYS` | `2` | Days of hotspot history (1–10) |
| `FIRMS_BBOX` | India region | `west,south,east,north` bounding box |
| `CONFIDENCE_THRESHOLD` | `0.45` | YOLOv8 detection confidence cutoff |
| `FRAME_SKIP` | `3` | Process every N-th frame (performance) |
| `MAX_CONCURRENT_STREAMS` | `10` | Max parallel video streams |
| `ALERT_WEBHOOK_URL` | — | Slack/Teams webhook for emergency alerts |

---

## 🗺 Severity Levels

| Level | FRP (Fire Radiative Power) | Colour |
|---|---|---|
| 🟡 Watch | < 100 MW | Yellow |
| 🟠 Warning | 100–499 MW | Orange |
| 🔴 Emergency | ≥ 500 MW | Red |

---

## 📄 License

MIT — free for academic and personal use.
