import os
import cv2
import numpy as np
import base64
import argparse
from pathlib import Path
from dotenv import load_dotenv
from inference_sdk import InferenceHTTPClient

load_dotenv()

CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=os.environ["ROBOFLOW_API_KEY"],
)

CLASS_CONFIG = {
    "fire": {"color": (0, 80, 255), "fill": (0, 40, 180)},
    "smoke": {"color": (180, 180, 180), "fill": (80, 80, 80)},
}


def draw_corner_box(frame, x1, y1, x2, y2, color, length=18):
    for p1, v, p2 in [
        ((x1, y1 + length), (x1, y1), (x1 + length, y1)),
        ((x2 - length, y1), (x2, y1), (x2, y1 + length)),
        ((x1, y2 - length), (x1, y2), (x1 + length, y2)),
        ((x2 - length, y2), (x2, y2), (x2, y2 - length)),
    ]:
        cv2.line(frame, p1, v, color, 2, cv2.LINE_AA)
        cv2.line(frame, v, p2, color, 2, cv2.LINE_AA)


def infer_frame(frame):
    small = cv2.resize(frame, (640, 640))
    _, buf = cv2.imencode(".jpg", small)
    b64 = base64.b64encode(buf).decode("utf-8")
    try:
        result = CLIENT.infer(b64, model_id="smoke-fire-wsde7/3")
        return result.get("predictions", [])
    except:
        return []


def draw(frame, preds):
    H, W = frame.shape[:2]
    alert = False
    for p in preds:
        cls = p["class"].lower()
        conf = p["confidence"]
        if cls not in CLASS_CONFIG:
            continue
        cfg = CLASS_CONFIG[cls]

        # roboflow returns center x,y,w,h
        cx, cy, pw, ph = p["x"], p["y"], p["width"], p["height"]
        sx, sy = W / 640, H / 640
        x1 = max(0, int((cx - pw / 2) * sx))
        y1 = max(0, int((cy - ph / 2) * sy))
        x2 = min(W - 1, int((cx + pw / 2) * sx))
        y2 = min(H - 1, int((cy + ph / 2) * sy))

        sub = frame[y1:y2, x1:x2]
        if sub.size > 0:
            ov = np.full(sub.shape, cfg["fill"], dtype=np.uint8)
            frame[y1:y2, x1:x2] = cv2.addWeighted(ov, 0.15, sub, 0.85, 0)

        draw_corner_box(frame, x1, y1, x2, y2, cfg["color"])

        label = f"{cls.capitalize()} {conf:.0%}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
        ly1 = max(0, y1 - th - 13)
        cv2.rectangle(frame, (x1, ly1), (x1 + tw + 20, y1), cfg["color"], -1)
        cv2.circle(frame, (x1 + 8, (ly1 + y1) // 2), 4, (255, 255, 255), -1)
        cv2.putText(
            frame,
            label,
            (x1 + 16, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        bw = int((x2 - x1) * conf)
        cv2.rectangle(frame, (x1, y2 + 4), (x2, y2 + 8), (30, 30, 30), -1)
        cv2.rectangle(frame, (x1, y2 + 4), (x1 + bw, y2 + 8), cfg["color"], -1)
        alert = True
    return frame, alert


def run(source, save):
    cap = cv2.VideoCapture(source)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    FPS = cap.get(cv2.CAP_PROP_FPS) or 30
    TOTAL = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"  {W}x{H} @ {FPS:.0f}fps | {TOTAL} frames")

    writer = None
    if save:
        Path("output").mkdir(exist_ok=True)
        out = f"output/{Path(source).stem}_detected.mp4"
        writer = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
        print(f"  Saving → {out}\n")

    frame_idx = 0
    last_preds = []
    SKIP = 5  # call API every 5th frame

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        if frame_idx % SKIP == 0:
            last_preds = infer_frame(frame)

        frame, alert = draw(frame, last_preds)

        status = "DETECTED" if alert else "ALL CLEAR"
        color = (0, 80, 255) if alert else (0, 200, 80)
        ov = frame.copy()
        cv2.rectangle(ov, (0, 0), (240, 50), (10, 10, 10), -1)
        cv2.addWeighted(ov, 0.6, frame, 0.4, 0, frame)
        cv2.rectangle(frame, (0, 0), (240, 2), color, -1)
        cv2.putText(
            frame,
            f"  {status}",
            (8, 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )

        prog = int(W * frame_idx / max(TOTAL, 1))
        cv2.rectangle(frame, (0, H - 3), (W, H), (25, 25, 25), -1)
        cv2.rectangle(frame, (0, H - 3), (prog, H), (0, 80, 200), -1)

        if writer:
            writer.write(frame)

        if frame_idx % 10 == 0:
            pct = frame_idx / max(TOTAL, 1) * 100
            bar = "█" * int(pct // 2) + "░" * (50 - int(pct // 2))
            print(f"\r  [{bar}] {pct:.0f}%", end="", flush=True)

    cap.release()
    if writer:
        writer.release()
    print(f"\n\n  Done → output/")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True)
    p.add_argument("--save", action="store_true")
    args = p.parse_args()
    run(args.source, args.save)
