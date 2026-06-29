from ultralytics import YOLO
import cv2
from pathlib import Path
import sys

VIDEO_PATH = sys.argv[1] if len(sys.argv) > 1 else "input/test.mp4"
OUTPUT_PATH = f"output/{Path(VIDEO_PATH).stem}_result.mp4"

CLASS_CONFIG = {
    0: {"name": "Fire",  "color": (0, 60, 255)},
    1: {"name": "Smoke", "color": (120, 120, 120)},
}

print(f"Loading model...")
model = YOLO("yolov8m.pt")  # downloads automatically, replace with models/best.pt after training

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"ERROR: Cannot open {VIDEO_PATH}")
    exit(1)

width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps    = cap.get(cv2.CAP_PROP_FPS)
total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"Video: {width}x{height} @ {fps:.1f}fps | {total} frames")

writer = cv2.VideoWriter(OUTPUT_PATH, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_idx += 1
    results = model(frame, verbose=False, conf=0.3)

    for result in results:
        if result.boxes is None:
            continue
        for box in result.boxes:
            cls  = int(box.cls[0])
            conf = float(box.conf[0])
            cfg  = CLASS_CONFIG.get(cls, {"name": f"cls{cls}", "color": (255,255,0)})
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            cv2.rectangle(frame, (x1, y1), (x2, y2), cfg["color"], 2)
            label = f"{cfg['name']} {conf:.0%}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 6, y1), cfg["color"], -1)
            cv2.putText(frame, label, (x1 + 3, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Progress bar in terminal
    pct = frame_idx / total * 100
    bar = "█" * int(pct // 2) + "░" * (50 - int(pct // 2))
    print(f"\r[{bar}] {pct:.1f}% ({frame_idx}/{total})", end="", flush=True)

    writer.write(frame)

cap.release()
writer.release()
print(f"\n✅ Done! Saved to {OUTPUT_PATH}")
