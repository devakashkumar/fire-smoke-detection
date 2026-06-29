import cv2
import asyncio
import threading
import time
from dataclasses import dataclass, field
from typing import Optional, Callable
from enum import Enum
import os

class SourceType(str, Enum):
    FILE = "file"
    RTSP = "rtsp"
    URL = "url"

class StreamStatus(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    ERROR = "error"
    STOPPED = "stopped"

@dataclass
class VideoSource:
    id: str
    name: str
    path: str
    source_type: SourceType
    location: Optional[str] = None
    lat: Optional[float] = None
    lng: Optional[float] = None
    status: StreamStatus = StreamStatus.IDLE
    fps: float = 0.0
    frame_count: int = 0
    error_msg: Optional[str] = None
    added_at: float = field(default_factory=time.time)

class StreamManager:
    def __init__(self):
        self.sources: dict[str, VideoSource] = {}
        self._captures: dict[str, cv2.VideoCapture] = {}
        self._threads: dict[str, threading.Thread] = {}
        self._stop_flags: dict[str, threading.Event] = {}
        self._frame_callbacks: dict[str, list[Callable]] = {}
        self._lock = threading.Lock()

    def add_source(self, source: VideoSource):
        with self._lock:
            self.sources[source.id] = source
            self._frame_callbacks[source.id] = []

    def remove_source(self, source_id: str):
        self.stop_stream(source_id)
        with self._lock:
            self.sources.pop(source_id, None)
            self._frame_callbacks.pop(source_id, None)

    def on_frame(self, source_id: str, callback: Callable):
        if source_id in self._frame_callbacks:
            self._frame_callbacks[source_id].append(callback)

    def start_stream(self, source_id: str):
        source = self.sources.get(source_id)
        if not source:
            return False
        stop_event = threading.Event()
        self._stop_flags[source_id] = stop_event
        t = threading.Thread(target=self._stream_loop, args=(source_id, stop_event), daemon=True)
        self._threads[source_id] = t
        source.status = StreamStatus.RUNNING
        t.start()
        return True

    def stop_stream(self, source_id: str):
        flag = self._stop_flags.get(source_id)
        if flag:
            flag.set()
        cap = self._captures.get(source_id)
        if cap:
            cap.release()
        source = self.sources.get(source_id)
        if source:
            source.status = StreamStatus.STOPPED

    def _stream_loop(self, source_id: str, stop_event: threading.Event):
        source = self.sources[source_id]
        cap = cv2.VideoCapture(source.path)
        if not cap.isOpened():
            source.status = StreamStatus.ERROR
            source.error_msg = f"Cannot open: {source.path}"
            return
        self._captures[source_id] = cap
        source.fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frame_skip = int(os.getenv("FRAME_SKIP", 3))
        frame_idx = 0
        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                if source.source_type == SourceType.FILE:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    source.status = StreamStatus.ERROR
                    break
            frame_idx += 1
            source.frame_count = frame_idx
            if frame_idx % frame_skip != 0:
                continue
            timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            for cb in self._frame_callbacks.get(source_id, []):
                try:
                    cb(source_id, frame, frame_idx, timestamp)
                except Exception as e:
                    print(f"[StreamManager] Callback error: {e}")
        cap.release()
        self._captures.pop(source_id, None)

    def get_all_sources(self) -> list[VideoSource]:
        return list(self.sources.values())

    def get_source(self, source_id: str) -> Optional[VideoSource]:
        return self.sources.get(source_id)
