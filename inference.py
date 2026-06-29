import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
from datetime import datetime
import argparse
import time
import logging
import os

# ── Logging setup ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/inference.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)

# ── Class config ───────────────────────────────────────────────────────────────
CLASS_CONFIG = {
    0: {"name": "Fire", "color": (0, 60, 255),   "emoji": "🔥"},
    1: {"name": "Smoke","color": (120, 120, 120), "emoji": "💨"},
}

def draw_detections(frame, results, conf_threshold=0.25):
    """Draw bounding boxes and labels on frame."""
    alert_triggered = False
    detection_summary = []

    for result in results:
        boxes = result.boxes
        if boxes is None:
            continue

        for box in boxes:
            conf = float(box.conf[0])
            cls  = int(box.cls[0])

            if conf < conf_threshold:
                continue

            cfg   = CLASS_CONFIG.get(cls, {"name": "Unknown", "color": (255,255,255), "emoji": "?"})
            color = cfg["color"]
            label = f"{cfg['emoji']} {cfg['name']} {conf:.1%}"

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Label background
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 8, y1), color, -1)
            cv2.putText(frame, label, (x1 + 4, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            alert_triggered = True
            detection_summary.append(f"{cfg['name']}({conf:.1%})")

    return frame, alert_triggered, detection_summary


def overlay_hud(frame, fps, alert, summary, frame_count):
    """Overlay HUD info on top-left of frame."""
    h, w = frame.shape[:2]
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Semi-transparent HUD bar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (350, 90), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    status_color = (0, 60, 255) if alert else (0, 220, 80)
    status_text  = "⚠ ALERT: " + ", ".join(summary) if alert else "✓ CLEAR"

    cv2.putText(frame, f"FPS: {fps:.1f}  Frame: {frame_count}", (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
    cv2.putText(frame, timestamp, (10, 44),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (180, 180, 180), 1)
    cv2.putText(frame, status_text, (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, status_color, 2)

    return frame


def run_on_image(model, source, conf, save_output):
    log.info(f"Running inference on image: {source}")
    frame = cv2.imread(source)
    if frame is None:
        log.error(f"Could not read image: {source}")
        return

    results = model(frame, verbose=False)
    frame, alert, summary = draw_detections(frame, results, conf)
    frame = overlay_hud(frame, 0.0, alert, summary, 1)

    if alert:
        log.warning(f"DETECTION: {', '.join(summary)}")
    else:
        log.info("No fire/smoke detected.")

    if save_output:
        out_path = f"output/{Path(source).stem}_result.jpg"
        cv2.imwrite(out_path, frame)
        log.info(f"Saved to {out_path}")

    cv2.imshow("Fire & Smoke Detection", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def run_on_video(model, source, conf, save_output):
    is_webcam = (source == "0" or source == 0)
    src = 0 if is_webcam else source
    log.info(f"Running inference on {'webcam' if is_webcam else f'video: {source}'}")

    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        log.error(f"Cannot open source: {source}")
        return

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30

    writer = None
    if save_output and not is_webcam:
        out_path = f"output/{Path(str(source)).stem}_result.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, fps_in, (width, height))
        log.info(f"Saving output video to {out_path}")

    frame_count = 0
    fps = 0.0
    prev_time = time.time()

    log.info("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        results = model(frame, verbose=False)
        frame, alert, summary = draw_detections(frame, results, conf)

        # FPS calc
        now = time.time()
        fps = 1.0 / (now - prev_time + 1e-9)
        prev_time = now

        frame = overlay_hud(frame, fps, alert, summary, frame_count)

        if alert:
            log.warning(f"Frame {frame_count} — DETECTION: {', '.join(summary)}")

        if writer:
            writer.write(frame)

        cv2.imshow("Fire & Smoke Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            log.info("User quit.")
            break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    log.info(f"Done. Processed {frame_count} frames.")


def main():
    parser = argparse.ArgumentParser(description="🔥💨 Fire & Smoke Detection Inference")
    parser.add_argument("--model",  default="models/best.pt",  help="Path to YOLOv8 .pt model")
    parser.add_argument("--source", default="0",               help="Image/video path or '0' for webcam")
    parser.add_argument("--conf",   default=0.25, type=float,  help="Confidence threshold (default: 0.25)")
    parser.add_argument("--save",   action="store_true",       help="Save output to /output folder")
    args = parser.parse_args()

    log.info("=" * 60)
    log.info("🔥💨  Fire & Smoke Detection — YOLOv8m Inference")
    log.info("=" * 60)
    log.info(f"Model  : {args.model}")
    log.info(f"Source : {args.source}")
    log.info(f"Conf   : {args.conf}")
    log.info(f"Save   : {args.save}")

    if not Path(args.model).exists():
        log.error(f"Model not found at: {args.model}")
        log.error("Place your YOLOv8 .pt file in the models/ folder and pass --model path.")
        return

    model = YOLO(args.model)
    log.info("Model loaded successfully.")

    source = args.source
    # Detect type
    img_exts   = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    video_exts = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    ext = Path(source).suffix.lower()

    if ext in img_exts:
        run_on_image(model, source, args.conf, args.save)
    else:
        run_on_video(model, source, args.conf, args.save)


if __name__ == "__main__":
    main()
