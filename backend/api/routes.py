"""
API routes — FastAPI router.
Bridges async WebSocket world ↔ threaded stream world via queue.Queue.
Exports: router, stream_manager, on_frame_received, queue_processor
"""

import asyncio
import base64
import json
import logging
import queue
import time
from typing import Optional

import cv2
from fastapi import APIRouter, File, HTTPException, Query, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from backend.core.detector import FireDetector
from backend.core.stream_manager import StreamManager, VideoSource, SourceType
from backend.services.alert_service import AlertService
from backend.services.firms_service import fetch_hotspots, get_summary

log = logging.getLogger("routes")

router         = APIRouter()
alert_svc      = AlertService()
stream_manager = StreamManager()
detector       = FireDetector()

# Thread-safe queue bridging stream threads → asyncio WebSocket broadcaster
frame_queue: queue.Queue = queue.Queue(maxsize=200)
clients: list[WebSocket] = []

# Camera lat/lng registry (source_id → {lat, lng, name, location})
camera_locations: dict[str, dict] = {}


# ── Called from stream threads (non-async) ───────────────────────────────────
def on_frame_received(source_id: str, frame, frame_idx: int, timestamp: float):
    """Callback registered with StreamManager. Runs in a thread."""
    try:
        annotated, detections, severity = detector.detect(frame)

        # Encode annotated frame to JPEG → base64
        _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 75])
        b64 = base64.b64encode(buf).decode()

        det_count = len(detections)
        confidence = max((d["confidence"] for d in detections), default=0.0)

        # Enqueue frame message
        msg = json.dumps({
            "type":            "frame",
            "source_id":       source_id,
            "frame":           b64,
            "severity":        severity,
            "detection_count": det_count,
            "timestamp":       time.time(),
        })
        try:
            frame_queue.put_nowait(msg)
        except queue.Full:
            try:
                frame_queue.get_nowait()
                frame_queue.put_nowait(msg)
            except queue.Empty:
                pass

        # Fire alert if threshold crossed
        source = stream_manager.get_source(source_id)
        src_name = source.name if source else source_id
        location = source.location if source else None

        if severity != "none" and alert_svc.should_alert(source_id, severity):
            alert = alert_svc.create_alert(
                source_id=source_id,
                source_name=src_name,
                severity=severity,
                confidence=confidence,
                frame_number=frame_idx,
                timestamp=timestamp,
                location=location,
            )
            alert_msg = json.dumps({
                "type":       "alert",
                "id":         alert.id,
                "source_id":  alert.source_id,
                "severity":   alert.severity,
                "confidence": alert.confidence,
                "timestamp":  alert.created_at,
            })
            try:
                frame_queue.put_nowait(alert_msg)
            except queue.Full:
                pass

    except Exception as e:
        log.error("on_frame_received error: %s", e)


# ── Background task: drain queue → broadcast to all WS clients ───────────────
async def queue_processor():
    loop = asyncio.get_event_loop()
    while True:
        try:
            msg = await loop.run_in_executor(None, lambda: frame_queue.get(timeout=0.05))
            dead = []
            for ws in list(clients):
                try:
                    await ws.send_text(msg)
                except Exception:
                    dead.append(ws)
            for ws in dead:
                if ws in clients:
                    clients.remove(ws)
        except queue.Empty:
            await asyncio.sleep(0.01)


# ── WebSocket ────────────────────────────────────────────────────────────────
@router.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    clients.append(ws)
    log.info("WS client connected (%d total)", len(clients))
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        if ws in clients:
            clients.remove(ws)
        log.info("WS client disconnected (%d remaining)", len(clients))


# ── Streams ──────────────────────────────────────────────────────────────────
class StreamPayload(BaseModel):
    url: str
    name: Optional[str] = None
    location: Optional[str] = None
    lat: Optional[float] = None
    lng: Optional[float] = None

@router.post("/streams")
async def add_stream(payload: StreamPayload):
    import uuid
    sid = f"rtsp-{uuid.uuid4().hex[:6]}"
    source = VideoSource(
        id=sid,
        name=payload.name or payload.url,
        path=payload.url,
        source_type=SourceType.RTSP,
        location=payload.location,
    )
    stream_manager.add_source(source)
    stream_manager.on_frame(sid, on_frame_received)
    stream_manager.start_stream(sid)
    if payload.lat is not None and payload.lng is not None:
        camera_locations[sid] = {
            "lat": payload.lat, "lng": payload.lng,
            "name": payload.name or sid,
            "location": payload.location or "",
        }
    return {"source_id": sid, "url": payload.url}

@router.get("/streams")
async def list_streams():
    sources = stream_manager.get_all_sources()
    return {"streams": [
        {"id": s.id, "name": s.name, "status": s.status,
         "source_type": s.source_type, "location": s.location}
        for s in sources
    ]}

@router.delete("/streams/{source_id}")
async def remove_stream(source_id: str):
    stream_manager.remove_source(source_id)
    return {"removed": source_id}


# ── Upload ───────────────────────────────────────────────────────────────────
@router.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    import shutil, pathlib, uuid
    dest = pathlib.Path("data/videos") / file.filename
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    sid = f"upload-{uuid.uuid4().hex[:6]}"
    source = VideoSource(
        id=sid,
        name=file.filename,
        path=str(dest),
        source_type=SourceType.FILE,
    )
    stream_manager.add_source(source)
    stream_manager.on_frame(sid, on_frame_received)
    stream_manager.start_stream(sid)
    return {"source_id": sid, "filename": file.filename}


# ── Cameras ──────────────────────────────────────────────────────────────────
@router.get("/cameras")
async def get_cameras():
    return {"cameras": camera_locations}


# ── Analytics ────────────────────────────────────────────────────────────────
@router.get("/analytics")
async def get_analytics():
    data = alert_svc.get_analytics()
    data["active_streams"] = len(stream_manager.get_all_sources())
    return data


# ── Alerts ───────────────────────────────────────────────────────────────────
@router.get("/alerts/export")
async def export_alerts():
    import csv, io
    from datetime import datetime
    from fastapi.responses import StreamingResponse
    alerts = alert_svc.get_all()
    out = io.StringIO()
    w = csv.writer(out)
    w.writerow(["ID","Source","Severity","Confidence%","Location","Timestamp","Acknowledged"])
    for a in alerts:
        ts = datetime.fromtimestamp(a.created_at).strftime("%Y-%m-%d %H:%M:%S")
        w.writerow([a.id, a.source_id, a.severity, f"{a.confidence*100:.1f}",
                    a.location or "", ts, "Yes" if a.acknowledged else "No"])
    out.seek(0)
    fname = f"fire_alerts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    return StreamingResponse(
        iter([out.getvalue()]), media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={fname}"}
    )

@router.get("/alerts")
async def get_alerts(limit: int = Query(50, le=200)):
    alerts = alert_svc.get_recent(limit)
    return {"alerts": [
        {"id": a.id, "source_id": a.source_id, "severity": a.severity,
         "confidence": a.confidence, "timestamp": a.created_at,
         "acknowledged": a.acknowledged, "location": a.location}
        for a in alerts
    ]}

@router.post("/alerts/{alert_id}/acknowledge")
async def acknowledge_alert(alert_id: str):
    ok = alert_svc.acknowledge(alert_id)
    if not ok:
        raise HTTPException(404, "Alert not found")
    return {"acknowledged": alert_id}


# ── NASA FIRMS ───────────────────────────────────────────────────────────────
@router.get("/firms")
async def get_firms(
    bbox:  str  = Query(None),
    days:  int  = Query(1, ge=1, le=10),
    force: bool = Query(False),
):
    from backend.services.firms_service import DEFAULT_BBOX
    hotspots = await fetch_hotspots(bbox=bbox or DEFAULT_BBOX, days=days, force=force)
    summary  = await get_summary()
    return {"hotspots": hotspots, "summary": summary, "count": len(hotspots)}


# ── Health ───────────────────────────────────────────────────────────────────
@router.get("/health")
async def health():
    return {
        "status":     "ok",
        "streams":    len(stream_manager.get_all_sources()),
        "ws_clients": len(clients),
        "queue_depth": frame_queue.qsize(),
    }