import cv2
import numpy as np
import time
import sys
import argparse
from pathlib import Path
from datetime import datetime

CLASS_CONFIG = {
    0: {
        "name": "Fire",
        "color_box":   (0,  80, 255),
        "color_fill":  (0,  40, 180),
        "color_glow":  (0,  60, 220),
        "dot_color":   (0, 120, 255),
        "icon": "F",
    },
    1: {
        "name": "Smoke",
        "color_box":   (180, 180, 180),
        "color_fill":  (100, 100, 100),
        "color_glow":  (140, 140, 140),
        "dot_color":   (200, 200, 200),
        "icon": "S",
    },
}

FIRE_ONLY_CLASSES = {0, 1}

def alpha_rect(frame, x1, y1, x2, y2, color, alpha=0.25):
    sub = frame[y1:y2, x1:x2]
    overlay = np.full(sub.shape, color, dtype=np.uint8)
    blended = cv2.addWeighted(overlay, alpha, sub, 1 - alpha, 0)
    frame[y1:y2, x1:x2] = blended

def draw_corner_box(frame, x1, y1, x2, y2, color, thickness=2, corner_len=18):
    corners = [
        ((x1, y1 + corner_len), (x1, y1), (x1 + corner_len, y1)),
        ((x2 - corner_len, y1), (x2, y1), (x2, y1 + corner_len)),
        ((x1, y2 - corner_len), (x1, y2), (x1 + corner_len, y2)),
        ((x2 - corner_len, y2), (x2, y2), (x2, y2 - corner_len)),
    ]
    for p1, vertex, p2 in corners:
        cv2.line(frame, p1, vertex, color, thickness, cv2.LINE_AA)
        cv2.line(frame, vertex, p2, color, thickness, cv2.LINE_AA)

def draw_scan_line(frame, x1, y1, x2, y2, color, progress):
    scan_y = int(y1 + (y2 - y1) * progress)
    scan_y = max(y1 + 1, min(y2 - 1, scan_y))
    overlay = frame.copy()
    cv2.line(overlay, (x1, scan_y), (x2, scan_y), color, 1, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

def draw_confidence_bar(frame, x1, y2, x2, conf, color):
    bar_h = 4
    bar_y = y2 + 6
    bar_x2 = x1 + int((x2 - x1) * conf)
    cv2.rectangle(frame, (x1, bar_y), (x2, bar_y + bar_h), (40, 40, 40), -1)
    cv2.rectangle(frame, (x1, bar_y), (bar_x2, bar_y + bar_h), color, -1)

def draw_detections(frame, results, conf_threshold, scan_progress):
    alert = False
    detections = []

    for result in results:
        if result.boxes is None:
            continue
        for box in result.boxes:
            cls  = int(box.cls[0])
            conf = float(box.conf[0])

            if cls not in FIRE_ONLY_CLASSES or conf < conf_threshold:
                continue

            cfg = CLASS_CONFIG[cls]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1]-1, x2), min(frame.shape[0]-1, y2)

            alpha_rect(frame, x1, y1, x2, y2, cfg["color_fill"], alpha=0.15)
            draw_scan_line(frame, x1, y1, x2, y2, cfg["color_glow"], scan_progress)
            draw_corner_box(frame, x1, y1, x2, y2, cfg["color_box"], thickness=2, corner_len=20)
            cv2.rectangle(frame, (x1, y1), (x2, y2), cfg["color_box"], 1)

            label = f"{cfg['name']}  {conf:.0%}"
            font       = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.52
            font_thick = 1
            (tw, th), _ = cv2.getTextSize(label, font, font_scale, font_thick)
            pad = 6
            lx1, ly1 = x1, y1 - th - pad * 2 - 1
            lx2, ly2 = x1 + tw + pad * 2 + 16, y1 - 1
            if ly1 < 0:
                ly1, ly2 = y2 + 1, y2 + th + pad * 2 + 1

            cv2.rectangle(frame, (lx1, ly1), (lx2, ly2), cfg["color_box"], -1)
            cv2.circle(frame, (lx1 + 8, (ly1 + ly2) // 2), 4, (255, 255, 255), -1)
            cv2.putText(frame, label, (lx1 + 16, ly2 - pad),
                        font, font_scale, (255, 255, 255), font_thick, cv2.LINE_AA)

            draw_confidence_bar(frame, x1, y2, x2, conf, cfg["color_box"])

            alert = True
            detections.append((cfg["name"], conf))

    return frame, alert, detections


def draw_hud(frame, fps, alert, detections, frame_idx, total_frames, source_name):
    h, w = frame.shape[:2]
    now  = datetime.now().strftime("%H:%M:%S")

    bar_w, bar_h = 300, 72
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (bar_w, bar_h), (10, 10, 10), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    accent_color = (0, 80, 255) if alert else (0, 200, 80)
    cv2.rectangle(frame, (0, 0), (bar_w, 2), accent_color, -1)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, f"FPS {fps:>5.1f}   {now}", (10, 20),
                font, 0.42, (160, 160, 160), 1, cv2.LINE_AA)
    short_src = Path(source_name).name[:30]
    cv2.putText(frame, short_src, (10, 38),
                font, 0.38, (100, 100, 100), 1, cv2.LINE_AA)

    if alert:
        det_str = "  |  ".join([f"{n} {c:.0%}" for n, c in detections])
        cv2.putText(frame, f"  DETECTED  {det_str}", (10, 60),
                    font, 0.44, (0, 100, 255), 1, cv2.LINE_AA)
    else:
        cv2.putText(frame, "  ALL CLEAR", (10, 60),
                    font, 0.44, (0, 200, 80), 1, cv2.LINE_AA)

    if total_frames > 0:
        pb_y  = h - 3
        filled = int(w * frame_idx / max(total_frames, 1))
        cv2.rectangle(frame, (0, pb_y), (w, h),      (25, 25, 25), -1)
        cv2.rectangle(frame, (0, pb_y), (filled, h), (0, 80, 200), -1)
        counter = f"{frame_idx}/{total_frames}"
        (cw, _), _ = cv2.getTextSize(counter, font, 0.38, 1)
        cv2.putText(frame, counter, (w - cw - 10, 16),
                    font, 0.38, (80, 80, 80), 1, cv2.LINE_AA)

    return frame


def run(source, conf, save, model_path):
    from ultralytics import YOLO
    print(f"\n  Loading model: {model_path}")
    model = YOLO(model_path)
    print(f"  Source: {source}  |  Conf: {conf}\n")

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open: {source}"); return

    W     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    FPS_IN = cap.get(cv2.CAP_PROP_FPS) or 30
    TOTAL  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"  {W}x{H} @ {FPS_IN:.1f}fps | {TOTAL} frames")

    writer = None
    if save:
        Path("output").mkdir(exist_ok=True)
        out_path = f"output/{Path(source).stem}_detected.mp4"
        writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), FPS_IN, (W, H))
        print(f"  Saving to: {out_path}")

    frame_idx, fps, scan_prog = 0, 0.0, 0.0
    t_prev = time.time()
    print("  Press 'q' to quit\n")

    while True:
        ret, frame = cap.read()
        if not ret: break

        frame_idx += 1
        scan_prog = (scan_prog + 0.04) % 1.0

        small = cv2.resize(frame, (320, 180))
        results = model(small, verbose=False, conf=conf, classes=list(FIRE_ONLY_CLASSES))
        frame, alert, detections = draw_detections(frame, results, conf, scan_prog)

        t_now  = time.time()
        fps    = 0.9 * fps + 0.1 * (1.0 / max(t_now - t_prev, 1e-6))
        t_prev = t_now

        frame = draw_hud(frame, fps, alert, detections, frame_idx, TOTAL, source)

        if writer:
            writer.write(frame)

        if frame_idx % 30 == 0:
            pct = frame_idx / max(TOTAL, 1) * 100
            bar = "█" * int(pct // 2) + "░" * (50 - int(pct // 2))
            print(f"\r  [{bar}] {pct:.0f}%", end="", flush=True)

    cap.release()
    if writer: writer.release()
    
    print(f"\n\n  Done. {frame_idx} frames processed.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True)
    p.add_argument("--model",  default="models/best.pt")
    p.add_argument("--conf",   default=0.30, type=float)
    p.add_argument("--save",   action="store_true")
    args = p.parse_args()
    run(args.source, args.conf, args.save, args.model)
