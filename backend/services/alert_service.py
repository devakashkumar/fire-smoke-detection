import sqlite3
import time
import httpx
import asyncio
import datetime
from dataclasses import dataclass, field
from typing import Optional
from collections import defaultdict
import os


@dataclass
class FireAlert:
    id: str
    source_id: str
    source_name: str
    severity: str
    confidence: float
    frame_number: int
    timestamp: float
    location: Optional[str] = None
    lat: Optional[float] = None
    lng: Optional[float] = None
    created_at: float = field(default_factory=time.time)
    acknowledged: bool = False


class AlertService:
    def __init__(self):
        db_url = os.getenv("DATABASE_URL", "sqlite:///./fire_detection.db")
        self._db_path = db_url.replace("sqlite:///./", "").replace("sqlite:///", "")
        self._conn = self._init_db()
        self.alerts: list[FireAlert] = self._load_from_db()
        self._last_alert_time: dict[str, float] = defaultdict(float)
        self._cooldown = 10.0
        self._webhook_url = os.getenv("ALERT_WEBHOOK_URL", "")
        self._subscribers: list = []

    def _init_db(self):
        conn = sqlite3.connect(self._db_path, check_same_thread=False)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS alerts (
                id TEXT PRIMARY KEY,
                source_id TEXT NOT NULL,
                source_name TEXT,
                severity TEXT NOT NULL,
                confidence REAL,
                frame_number INTEGER,
                timestamp REAL,
                location TEXT,
                lat REAL,
                lng REAL,
                created_at REAL NOT NULL,
                acknowledged INTEGER DEFAULT 0
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_created_at ON alerts(created_at)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_source_id  ON alerts(source_id)")
        conn.commit()
        return conn

    def _load_from_db(self) -> list[FireAlert]:
        rows = self._conn.execute(
            "SELECT id,source_id,source_name,severity,confidence,frame_number,"
            "timestamp,location,lat,lng,created_at,acknowledged "
            "FROM alerts ORDER BY created_at DESC LIMIT 500"
        ).fetchall()
        return [
            FireAlert(
                id=r[0], source_id=r[1], source_name=r[2] or r[1],
                severity=r[3], confidence=r[4] or 0.0,
                frame_number=r[5] or 0, timestamp=r[6] or 0.0,
                location=r[7], lat=r[8], lng=r[9],
                created_at=r[10], acknowledged=bool(r[11])
            )
            for r in rows
        ]

    def _persist(self, alert: FireAlert):
        self._conn.execute("""
            INSERT OR REPLACE INTO alerts
            (id,source_id,source_name,severity,confidence,frame_number,
             timestamp,location,lat,lng,created_at,acknowledged)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            alert.id, alert.source_id, alert.source_name,
            alert.severity, alert.confidence, alert.frame_number,
            alert.timestamp, alert.location, alert.lat, alert.lng,
            alert.created_at, int(alert.acknowledged)
        ))
        self._conn.commit()

    def subscribe(self, callback):
        self._subscribers.append(callback)

    def should_alert(self, source_id: str, severity: str) -> bool:
        if severity == "none":
            return False
        now = time.time()
        key = f"{source_id}:{severity}"
        if now - self._last_alert_time[key] < self._cooldown:
            return False
        self._last_alert_time[key] = now
        return True

    def create_alert(
        self,
        source_id: str,
        source_name: str,
        severity: str,
        confidence: float,
        frame_number: int,
        timestamp: float,
        location: Optional[str] = None,
        lat: Optional[float] = None,
        lng: Optional[float] = None,
    ) -> FireAlert:
        alert = FireAlert(
            id=f"{source_id}-{int(time.time() * 1000)}",
            source_id=source_id,
            source_name=source_name,
            severity=severity,
            confidence=confidence,
            frame_number=frame_number,
            timestamp=timestamp,
            location=location,
            lat=lat,
            lng=lng,
        )
        self.alerts.insert(0, alert)
        if len(self.alerts) > 500:
            self.alerts = self.alerts[:500]
        self._persist(alert)
        for cb in self._subscribers:
            try:
                cb(alert)
            except Exception as e:
                print(f"[AlertService] Subscriber error: {e}")
        if self._webhook_url:
            asyncio.create_task(self._send_webhook(alert))
        return alert

    async def _send_webhook(self, alert: FireAlert):
        try:
            async with httpx.AsyncClient() as client:
                await client.post(
                    self._webhook_url,
                    json={
                        "severity": alert.severity,
                        "source": alert.source_name,
                        "confidence": alert.confidence,
                        "location": alert.location,
                        "time": alert.created_at,
                    },
                    timeout=5,
                )
        except Exception as e:
            print(f"[AlertService] Webhook error: {e}")

    def acknowledge(self, alert_id: str) -> bool:
        for a in self.alerts:
            if a.id == alert_id:
                a.acknowledged = True
                self._conn.execute(
                    "UPDATE alerts SET acknowledged=1 WHERE id=?", (alert_id,)
                )
                self._conn.commit()
                return True
        return False

    def get_recent(self, limit: int = 50) -> list[FireAlert]:
        return self.alerts[:limit]

    def get_all(self) -> list[FireAlert]:
        return self.alerts

    def get_stats(self) -> dict:
        by_severity: dict[str, int] = defaultdict(int)
        for a in self.alerts:
            by_severity[a.severity] += 1
        return {"total": len(self.alerts), "by_severity": dict(by_severity)}

    def get_analytics(self) -> dict:
        now = time.time()
        alerts = self.alerts

        # ── Hourly breakdown — last 24 h ──────────────────────────
        hourly_labels, hourly_emg, hourly_wrn, hourly_wch = [], [], [], []
        for h in range(23, -1, -1):          # oldest → newest
            start = now - (h + 1) * 3600
            end   = now - h * 3600
            bucket = [a for a in alerts if start <= a.created_at < end]
            hourly_labels.append(
                datetime.datetime.fromtimestamp(end).strftime("%H:00")
            )
            hourly_emg.append(sum(1 for a in bucket if a.severity == "emergency"))
            hourly_wrn.append(sum(1 for a in bucket if a.severity == "warning"))
            hourly_wch.append(sum(1 for a in bucket if a.severity == "watch"))

        # ── Daily breakdown — last 7 days ─────────────────────────
        daily_labels, daily_counts = [], []
        for d in range(6, -1, -1):
            start = now - (d + 1) * 86400
            end   = now - d * 86400
            daily_labels.append(
                datetime.datetime.fromtimestamp(end).strftime("%b %d")
            )
            daily_counts.append(sum(1 for a in alerts if start <= a.created_at < end))

        # ── Severity totals ───────────────────────────────────────
        by_severity = {"emergency": 0, "warning": 0, "watch": 0}
        for a in alerts:
            if a.severity in by_severity:
                by_severity[a.severity] += 1

        # ── Per-source totals ─────────────────────────────────────
        by_source: dict[str, int] = defaultdict(int)
        for a in alerts:
            by_source[a.source_id] += 1

        unacknowledged = sum(1 for a in alerts if not a.acknowledged)

        return {
            "total":          len(alerts),
            "unacknowledged": unacknowledged,
            "by_severity":    by_severity,
            "by_source":      dict(by_source),
            "hourly": {
                "labels":    hourly_labels,
                "emergency": hourly_emg,
                "warning":   hourly_wrn,
                "watch":     hourly_wch,
            },
            "daily": {
                "labels": daily_labels,
                "counts": daily_counts,
            },
        }
