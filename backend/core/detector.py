"""
FireDetector — YOLOv8 inference with OpenCV annotation.
Draws filled+bordered rectangles, confidence labels, severity overlays.
Supports generic COCO model fallback and optional fire-specific model.
"""

import cv2
import numpy as np
import os
from pathlib import Path
from typing import Optional

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

# Severity thresholds (confidence-based)
SEVERITY_THRESHOLDS = {
    "emergency": 0.80,
    "warning":   0.60,
    "watch":     0.40,
}

# BGR colors for severity tiers
SEVERITY_COLORS = {
    "emergency": (0,   40,  220),   # vivid red
    "warning":   (0,  140,  255),   # amber
    "watch":     (0,  210,  255),   # yellow
    "none":      (80,  80,   80),
}

# Label text colors (white always readable)
LABEL_BG = {
    "emergency": (0,   30,  180),
    "warning":   (0,  100,  200),
    "watch":     (0,  160,  210),
}

CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.45"))
FRAME_SKIP           = int(os.getenv("FRAME_SKIP", "3"))


def _classify_severity(confidence: float) -> str:
    if confidence >= SEVERITY_THRESHOLDS["emergency"]:
        return "emergency"
    if confidence >= SEVERITY_THRESHOLDS["warning"]:
        return "warning"
    if confidence >= SEVERITY_THRESHOLDS["watch"]:
        return "watch"
    return "watch"


def _draw_box(frame: np.ndarray, x1: int, y1: int, x2: int, y2: int,
              confidence: float, severity: str, label: str = "FIRE") -> np.ndarray:
    """Draw a severity-colored bounding box with label on the frame."""
    color  = SEVERITY_COLORS[severity]
    lbg    = LABEL_BG.get(severity, (30, 30, 30))
    thick  = 3 if severity == "emergency" else 2

    # --- main rectangle ---
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thick)

    # --- corner ticks (tactical look) ---
    tick = 12
    for (cx, cy) in [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]:
        dx = 1 if cx == x1 else -1
        dy = 1 if cy == y1 else -1
        cv2.line(frame, (cx, cy), (cx + dx * tick, cy), color, thick + 1)
        cv2.line(frame, (cx, cy), (cx, cy + dy * tick), color, thick + 1)

    # --- label pill ---
    tag_text = f"{label}  {confidence:.0%}  {severity.upper()}"
    font     = cv2.FONT_HERSHEY_SIMPLEX
    scale    = 0.52
    (tw, th), baseline = cv2.getTextSize(tag_text, font, scale, 1)
    pad = 5
    lx1, ly1 = x1, max(0, y1 - th - pad * 2 - baseline)
    lx2, ly2 = x1 + tw + pad * 2, y1

    # semi-transparent pill background
    overlay = frame.copy()
    cv2.rectangle(overlay, (lx1, ly1), (lx2, ly2), lbg, -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    cv2.putText(frame, tag_text,
                (lx1 + pad, ly2 - baseline - 1),
                font, scale, (255, 255, 255), 1, cv2.LINE_AA)

    return frame


def _hsv_fire_detect(frame: np.ndarray) -> list[dict]:
    """HSV color-space fallback: detect orange/red fire blobs."""
    hsv   = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, np.array([0,  120,  80]), np.array([15, 255, 255]))
    mask2 = cv2.inRange(hsv, np.array([160,120,  80]), np.array([180,255, 255]))
    mask  = cv2.bitwise_or(mask1, mask2)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detections  = []
    h, w = frame.shape[:2]
    min_area = (w * h) * 0.003   # at least 0.3 % of frame

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        bx, by, bw, bh = cv2.boundingRect(cnt)
        conf = min(0.55, 0.35 + (area / (w * h)) * 5)   # synthetic confidence
        detections.append({
            "bbox":       (bx, by, bx + bw, by + bh),
            "confidence": conf,
            "class":      "fire",
            "source":     "hsv",
        })
    return detections


class FireDetector:
    """
    Wraps YOLOv8 (COCO or fire-specific) + HSV fallback.
    Call detect(frame) → annotated frame, list[dict], severity str.
    """

    def __init__(self):
        self.model        = None
        self.model_type   = "none"
        self._frame_count = 0
        self._load_model()

    # ------------------------------------------------------------------
    def _load_model(self):
        if not YOLO_AVAILABLE:
            print("[detector] ultralytics not installed — HSV-only mode")
            return

        # Priority 1: fire-specific model
        for candidate in ["fire_model.pt", "yolov8_fire.pt", "fire_smoke.pt"]:
            p = Path(candidate)
            if p.exists():
                self.model      = YOLO(str(p))
                self.model_type = "fire_specific"
                print(f"[detector] Loaded fire-specific model: {p}")
                return

        # Priority 2: COCO YOLOv8n (generic — no fire class, HSV supplements)
        try:
            self.model      = YOLO("yolov8n.pt")
            self.model_type = "coco"
            print("[detector] Loaded YOLOv8n (COCO). "
                  "Download fire_model.pt for better accuracy.")
        except Exception as e:
            print(f"[detector] YOLO load failed: {e} — HSV-only mode")

    # ------------------------------------------------------------------
    def detect(self, frame: np.ndarray) -> tuple[np.ndarray, list[dict], str]:
        """
        Returns:
            annotated_frame  — frame with boxes drawn
            detections       — list of detection dicts
            overall_severity — 'emergency' | 'warning' | 'watch' | 'none'
        """
        self._frame_count += 1
        annotated = frame.copy()
        all_detections: list[dict] = []

        # --- YOLO inference ---
        if self.model is not None:
            results = self.model(frame, verbose=False, conf=CONFIDENCE_THRESHOLD)
            for r in results:
                if r.boxes is None:
                    continue
                for box in r.boxes:
                    conf  = float(box.conf[0])
                    cls   = int(box.cls[0])
                    label = self.model.names[cls].lower() if self.model_type == "coco" else "fire"

                    # For fire-specific models accept everything
                    # For COCO only proxy: class 76 = cell phone skipped, look for 0 person etc
                    # Real COCO has no "fire" class — rely on HSV for COCO
                    if self.model_type == "fire_specific" or label in ("fire", "smoke", "wildfire"):
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        all_detections.append({
                            "bbox":       (x1, y1, x2, y2),
                            "confidence": conf,
                            "class":      label,
                            "source":     "yolo",
                        })

        # --- HSV fallback (always run on COCO or no-model) ---
        if self.model_type != "fire_specific":
            hsv_dets = _hsv_fire_detect(frame)
            all_detections.extend(hsv_dets)

        # --- Annotate frame ---
        overall_severity = "none"
        for det in all_detections:
            sev = _classify_severity(det["confidence"])
            det["severity"] = sev

            # Track highest severity
            order = ["none", "watch", "warning", "emergency"]
            if order.index(sev) > order.index(overall_severity):
                overall_severity = sev

            x1, y1, x2, y2 = det["bbox"]
            _draw_box(annotated, x1, y1, x2, y2,
                      det["confidence"], sev, det["class"].upper())

        # --- Timestamp + status HUD ---
        self._draw_hud(annotated, len(all_detections), overall_severity)

        return annotated, all_detections, overall_severity

    # ------------------------------------------------------------------
    @staticmethod
    def _draw_hud(frame: np.ndarray, det_count: int, severity: str):
        """Bottom-left HUD: detection count + severity status."""
        import datetime
        h, w = frame.shape[:2]
        ts    = datetime.datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
        font  = cv2.FONT_HERSHEY_SIMPLEX

        lines = [
            f"DETECTIONS: {det_count}",
            f"STATUS: {severity.upper()}",
            ts,
        ]
        y0 = h - 10 - len(lines) * 22
        for i, txt in enumerate(lines):
            y = y0 + i * 22
            color = SEVERITY_COLORS.get(severity, (180, 180, 180)) if i == 1 else (180, 180, 180)
            cv2.putText(frame, txt, (10, y), font, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(frame, txt, (10, y), font, 0.45, color,     1, cv2.LINE_AA)