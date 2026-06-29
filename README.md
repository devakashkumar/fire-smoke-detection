# 🔥💨 Fire & Smoke Detection — YOLOv8

Real-time fire and smoke detection using **YOLOv8** on video files, webcam streams, and Roboflow cloud inference.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-orange?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

---

## ✨ Scripts

| Script | Description |
|---|---|
| `inference.py` | Full-featured inference on image or video with HUD overlay |
| `fast_detect.py` | Optimised video inference — runs model every 3rd frame for speed |
| `video_detect.py` | Premium visual output with corner boxes, scan lines & confidence bars |
| `rf_detect.py` | Cloud inference via **Roboflow API** (no local GPU required) |
| `test_pipeline.py` | Quick sanity-check pipeline on a test video |
| `download_model.py` | Downloads YOLOv8m base weights as a placeholder for `models/best.pt` |

---

## 🚀 Quick Start

### 1. Clone & set up environment

```bash
git clone https://github.com/YOURUSERNAME/fire-smoke-detection.git
cd fire-smoke-detection

python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Get a model

**Option A — Use your own trained weights:**
```bash
# Drop your trained YOLOv8 .pt file into models/
cp path/to/your/best.pt models/best.pt
```

**Option B — Download a base YOLOv8 model (placeholder):**
```bash
python download_model.py
```
> Note: the base model is COCO-pretrained and won't reliably detect fire/smoke.
> Train on the [Roboflow dataset](https://universe.roboflow.com/firesmokedataset/smoke-fire-wsde7) for best results.

### 3. Run detection

```bash
# On a video file
python inference.py --source input/fire.mp4 --model models/best.pt --save

# On webcam
python inference.py --source 0 --model models/best.pt

# Fast mode (every 3rd frame)
python fast_detect.py --source input/fire.mp4 --model models/best.pt --save

# Premium visuals
python video_detect.py --source input/fire.mp4 --model models/best.pt --save

# Roboflow cloud inference (no local model needed)
cp .env.example .env   # then set RF_API_KEY in .env
python rf_detect.py --source input/fire.mp4
```

---

## 📁 Project Structure

```
fire-smoke-detection/
├── inference.py          # Full-featured YOLOv8 inference (image + video)
├── fast_detect.py        # Speed-optimised video detection
├── video_detect.py       # Premium visual overlay detection
├── rf_detect.py          # Roboflow cloud inference
├── test_pipeline.py      # Quick sanity-check pipeline
├── download_model.py     # Helper: download base YOLOv8 weights
├── FIRE_DATASET_URL      # Dataset reference (Roboflow)
├── requirements.txt
├── .env.example          # Environment variable template
├── models/               # ← drop your .pt files here (gitignored)
├── input/                # ← drop input videos here (gitignored)
└── output/               # Detection results saved here (gitignored)
```

---

## ⚙️ CLI Reference

### `inference.py` / `fast_detect.py` / `video_detect.py`

| Argument | Default | Description |
|---|---|---|
| `--source` | `0` | Path to video/image or `0` for webcam |
| `--model` | `models/best.pt` | Path to YOLOv8 `.pt` model |
| `--conf` | `0.25` | Confidence threshold |
| `--save` | flag | Save annotated output to `output/` |

### `rf_detect.py` (Roboflow cloud)

| Argument | Default | Description |
|---|---|---|
| `--source` | required | Path to video file |
| `--save` | flag | Save annotated output to `output/` |

> Requires `RF_API_KEY` in `.env` or exported in your shell.

---

## 🏷 Classes

| ID | Class | Colour |
|---|---|---|
| 0 | 🔥 Fire | Blue-red `(0, 80, 255)` |
| 1 | 💨 Smoke | Grey `(180, 180, 180)` |

---

## 📄 Dataset & Training

- **Dataset**: [Smoke & Fire — Roboflow Universe](https://universe.roboflow.com/firesmokedataset/smoke-fire-wsde7)
- **Base model**: `yolov8n.pt` (Ultralytics)
- **Training**: 30 epochs, 416×416, batch 8, CPU

---

## 📄 License

MIT — free for academic and personal use.
