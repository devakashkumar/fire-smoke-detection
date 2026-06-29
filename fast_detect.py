import cv2
import numpy as np
import time
import argparse
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO

CLASS_CONFIG = {
    0: {"name": "Fire",  "color": (0, 80, 255),   "fill": (0, 40, 180)},
    1: {"name": "Smoke", "color": (180, 180, 180), "fill": (100, 100, 100)},
}

def draw_corner_box(frame, x1, y1, x2, y2, color, length=18):
    for p1, v, p2 in [
        ((x1, y1+length),(x1,y1),(x1+length,y1)),
        ((x2-length,y1),(x2,y1),(x2,y1+length)),
        ((x1,y2-length),(x1,y2),(x1+length,y2)),
        ((x2-length,y2),(x2,y2),(x2,y2-length)),
    ]:
        cv2.line(frame, p1, v, color, 2, cv2.LINE_AA)
        cv2.line(frame, v, p2, color, 2, cv2.LINE_AA)

def run(source, conf, save, model_path):
    print(f"\n  Loading model...")
    model = YOLO(model_path)

    cap = cv2.VideoCapture(source)
    W     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    FPS   = cap.get(cv2.CAP_PROP_FPS) or 30
    TOTAL = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"  {W}x{H} @ {FPS:.0f}fps | {TOTAL} frames")

    writer = None
    if save:
        Path("output").mkdir(exist_ok=True)
        out = f"output/{Path(source).stem}_detected.mp4"
        writer = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
        print(f"  Saving → {out}\n")

    frame_idx = 0
    last_results = []
    SKIP = 3  # run model every 3rd frame, reuse results in between

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        if frame_idx % SKIP == 0:
            small = cv2.resize(frame, (416, 416))
            last_results = model(small, verbose=False, conf=conf, imgsz=416)

        sx = W / 416
        sy = H / 416

        alert = False
        for result in last_results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                cls  = int(box.cls[0])
                conf_val = float(box.conf[0])
                if cls not in CLASS_CONFIG:
                    continue
                cfg = CLASS_CONFIG[cls]
                bx1, by1, bx2, by2 = box.xyxy[0]
                x1 = max(0, int(bx1 * sx))
                y1 = max(0, int(by1 * sy))
                x2 = min(W-1, int(bx2 * sx))
                y2 = min(H-1, int(by2 * sy))

                # translucent fill
                sub = frame[y1:y2, x1:x2]
                if sub.size > 0:
                    overlay = np.full(sub.shape, cfg["fill"], dtype=np.uint8)
                    frame[y1:y2, x1:x2] = cv2.addWeighted(overlay, 0.15, sub, 0.85, 0)

                draw_corner_box(frame, x1, y1, x2, y2, cfg["color"])

                label = f"{cfg['name']} {conf_val:.0%}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
                ly1 = max(0, y1 - th - 13)
                cv2.rectangle(frame, (x1, ly1), (x1 + tw + 20, y1), cfg["color"], -1)
                cv2.circle(frame, (x1 + 8, (ly1 + y1) // 2), 4, (255,255,255), -1)
                cv2.putText(frame, label, (x1+16, y1-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255,255,255), 1, cv2.LINE_AA)

                bw = int((x2-x1) * conf_val)
                cv2.rectangle(frame, (x1, y2+4), (x2, y2+8), (30,30,30), -1)
                cv2.rectangle(frame, (x1, y2+4), (x1+bw, y2+8), cfg["color"], -1)
                alert = True

        # HUD
        status = "DETECTED" if alert else "ALL CLEAR"
        color  = (0,80,255) if alert else (0,200,80)
        ov = frame.copy()
        cv2.rectangle(ov, (0,0), (260, 52), (10,10,10), -1)
        cv2.addWeighted(ov, 0.6, frame, 0.4, 0, frame)
        cv2.rectangle(frame, (0,0), (260,2), color, -1)
        cv2.putText(frame, f"FPS target:{FPS:.0f}  skip:{SKIP}", (8,18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (120,120,120), 1)
        cv2.putText(frame, f"  {status}", (8,42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)

        # progress
        prog = int(W * frame_idx / max(TOTAL,1))
        cv2.rectangle(frame, (0, H-3), (W, H), (25,25,25), -1)
        cv2.rectangle(frame, (0, H-3), (prog, H), (0,80,200), -1)

        if writer:
            writer.write(frame)

        if frame_idx % 30 == 0:
            pct = frame_idx / max(TOTAL,1) * 100
            bar = "█" * int(pct//2) + "░" * (50-int(pct//2))
            print(f"\r  [{bar}] {pct:.0f}%  frame {frame_idx}/{TOTAL}", end="", flush=True)

    cap.release()
    if writer:
        writer.release()
    print(f"\n\n  Done. Saved to output/")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True)
    p.add_argument("--model",  default="models/best.pt")
    p.add_argument("--conf",   default=0.25, type=float)
    p.add_argument("--save",   action="store_true")
    args = p.parse_args()
    run(args.source, args.conf, args.save, args.model)
