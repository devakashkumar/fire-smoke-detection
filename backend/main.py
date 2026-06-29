import os
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from backend.api.routes import router, stream_manager, on_frame_received, queue_processor
from backend.core.stream_manager import VideoSource, SourceType
import asyncio

app = FastAPI(title="Forest Fire Detection System", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.include_router(router, prefix="/api")
app.mount("/static", StaticFiles(directory="frontend"), name="static")

@app.get("/")
async def serve_dashboard():
    return FileResponse("frontend/index.html")

@app.on_event("startup")
async def startup():
    asyncio.create_task(queue_processor())
    video_dir = os.getenv("VIDEO_DIR", "data/videos")
    skip = {"fire_sample.mp4"}
    videos = sorted([f for f in os.listdir(video_dir)
                     if f.endswith((".mp4", ".avi", ".mov")) and f not in skip])
    for i, v in enumerate(videos):
        path = os.path.join(video_dir, v)
        sid = f"auto-{i+1}"
        source = VideoSource(id=sid, name=v, path=path,
                             source_type=SourceType.FILE, location=f"Camera {i+1}")
        stream_manager.add_source(source)
        stream_manager.on_frame(sid, on_frame_received)
        stream_manager.start_stream(sid)
        print(f"[Startup] Loaded: {v} → {sid}")

@app.get("/health")
async def health():
    return {"status": "ok"}
