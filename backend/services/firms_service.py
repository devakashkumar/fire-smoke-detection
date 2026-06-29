"""
NASA FIRMS (Fire Information for Resource Management System) integration.

Fetches real-time active fire hotspots from VIIRS satellite data.
Public API — no key required for the basic area endpoint.
Docs: https://firms.modaps.eosdis.nasa.gov/api/
"""

import asyncio
import csv
import io
import logging
import os
import time
from dataclasses import dataclass, asdict
from typing import Optional

import httpx

log = logging.getLogger("firms")

# Public FIRMS MAP_KEY (no-auth, rate limited to ~10 req/day per IP)
# Replace with a real key from https://firms.modaps.eosdis.nasa.gov/api/map_key/
FIRMS_MAP_KEY = os.getenv("FIRMS_MAP_KEY", "FIRMS_MAP_KEY")

# Default bounding box: India + adjacent high-risk zones
# Format: west,south,east,north
DEFAULT_BBOX = os.getenv("FIRMS_BBOX", "68.0,8.0,97.0,37.0")

# Available satellite sources
SOURCES = {
    "VIIRS_SNPP_NRT": "VIIRS SNPP Near Real-Time",
    "MODIS_NRT":      "MODIS Near Real-Time",
    "VIIRS_NOAA20_NRT": "VIIRS NOAA-20 Near Real-Time",
}

DEFAULT_SOURCE = os.getenv("FIRMS_SOURCE", "VIIRS_SNPP_NRT")
DEFAULT_DAYS   = int(os.getenv("FIRMS_DAYS", "1"))   # 1 = last 24 hours

# Cache to avoid hammering the API
_cache: dict = {"data": [], "ts": 0, "bbox": ""}
CACHE_TTL = 900   # 15 minutes


@dataclass
class FireHotspot:
    latitude:    float
    longitude:   float
    brightness:  float          # Kelvin (VIIRS) or K (MODIS)
    scan:        float          # scan size km
    track:       float          # track size km
    acq_date:    str            # YYYY-MM-DD
    acq_time:    str            # HHMM UTC
    satellite:   str
    confidence:  str            # 'l' | 'n' | 'h'  or numeric 0-100
    frp:         float          # Fire Radiative Power (MW)
    daynight:    str            # 'D' | 'N'
    source:      str            # which satellite product

    @property
    def confidence_label(self) -> str:
        mapping = {"l": "Low", "n": "Nominal", "h": "High"}
        return mapping.get(str(self.confidence).lower(), str(self.confidence))

    @property
    def severity(self) -> str:
        """Map FRP to rough severity tier."""
        if self.frp >= 500:
            return "emergency"
        if self.frp >= 100:
            return "warning"
        return "watch"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["confidence_label"] = self.confidence_label
        d["severity"]         = self.severity
        return d


def _parse_csv(raw: str, source: str) -> list[FireHotspot]:
    hotspots = []
    reader   = csv.DictReader(io.StringIO(raw.strip()))
    for row in reader:
        try:
            hotspots.append(FireHotspot(
                latitude   = float(row.get("latitude",  row.get("lat",  0))),
                longitude  = float(row.get("longitude", row.get("lon",  0))),
                brightness = float(row.get("bright_ti4", row.get("brightness", 0)) or 0),
                scan       = float(row.get("scan",  0) or 0),
                track      = float(row.get("track", 0) or 0),
                acq_date   = row.get("acq_date", ""),
                acq_time   = row.get("acq_time", ""),
                satellite  = row.get("satellite", ""),
                confidence = row.get("confidence", "n"),
                frp        = float(row.get("frp", 0) or 0),
                daynight   = row.get("daynight", "D"),
                source     = source,
            ))
        except (ValueError, KeyError):
            continue
    return hotspots


async def fetch_hotspots(
    bbox:   str = DEFAULT_BBOX,
    source: str = DEFAULT_SOURCE,
    days:   int = DEFAULT_DAYS,
    force:  bool = False,
) -> list[dict]:
    """
    Fetch active fire hotspots from NASA FIRMS.
    Results are cached for 15 min to respect rate limits.
    Returns a list of dicts ready for JSON serialisation.
    """
    global _cache

    now = time.time()
    if (not force
            and _cache["data"]
            and _cache["bbox"] == bbox
            and (now - _cache["ts"]) < CACHE_TTL):
        log.debug("FIRMS cache hit (%d hotspots)", len(_cache["data"]))
        return _cache["data"]

    url = (
        f"https://firms.modaps.eosdis.nasa.gov/api/area/csv"
        f"/{FIRMS_MAP_KEY}/{source}/{bbox}/{days}"
    )

    log.info("Fetching FIRMS data: %s", url)

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            raw = resp.text
    except httpx.HTTPStatusError as e:
        log.error("FIRMS HTTP error %s: %s", e.response.status_code, e.response.text[:200])
        if e.response.status_code == 400 and FIRMS_MAP_KEY == "FIRMS_MAP_KEY":
            log.warning("FIRMS_MAP_KEY not set — get a free key at "
                        "https://firms.modaps.eosdis.nasa.gov/api/map_key/")
        # Return stale cache on error if available
        return _cache["data"]
    except Exception as e:
        log.error("FIRMS fetch failed: %s", e)
        return _cache["data"]

    hotspots = _parse_csv(raw, source)
    result   = [h.to_dict() for h in hotspots]

    _cache = {"data": result, "ts": now, "bbox": bbox}
    log.info("FIRMS: fetched %d hotspots", len(result))
    return result


async def get_summary() -> dict:
    """High-level summary for the dashboard status bar."""
    hotspots = await fetch_hotspots()
    if not hotspots:
        return {"total": 0, "emergency": 0, "warning": 0, "watch": 0, "source": DEFAULT_SOURCE}

    counts: dict[str, int] = {"emergency": 0, "warning": 0, "watch": 0}
    for h in hotspots:
        counts[h["severity"]] = counts.get(h["severity"], 0) + 1

    return {
        "total":     len(hotspots),
        "emergency": counts["emergency"],
        "warning":   counts["warning"],
        "watch":     counts["watch"],
        "source":    SOURCES.get(DEFAULT_SOURCE, DEFAULT_SOURCE),
        "bbox":      DEFAULT_BBOX,
    }