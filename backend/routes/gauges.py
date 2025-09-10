from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional

from models import (
    init_models,
    upsert_gauge,
    set_gauge_calibration,
    set_gauge_thresholds,
    recent_gauge_readings,
    recent_events,
)


router = APIRouter()


class GaugeIn(BaseModel):
    name: str
    camera_id: str
    roi_x0: Optional[int] = None
    roi_y0: Optional[int] = None
    roi_x1: Optional[int] = None
    roi_y1: Optional[int] = None
    min_angle: Optional[float] = None
    max_angle: Optional[float] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    gauge_id: Optional[int] = None


class CalibrationIn(BaseModel):
    min_angle: float
    max_angle: float
    min_value: float
    max_value: float


class ThresholdsIn(BaseModel):
    low_warn: float
    high_warn: float
    low_crit: float
    high_crit: float
    roc_limit: float
    cusum_k: float = 0.2
    cusum_h: float = 3.0


@router.post("/gauges")
def create_or_update_gauge(payload: GaugeIn):
    from main import DB_PATH  # reuse same DB
    init_models(DB_PATH)
    roi = None
    if payload.roi_x0 is not None:
        roi = (payload.roi_x0 or 0, payload.roi_y0 or 0, payload.roi_x1 or 0, payload.roi_y1 or 0)
    cal = None
    if payload.min_angle is not None:
        cal = {
            "min_angle": payload.min_angle,
            "max_angle": payload.max_angle,
            "min_value": payload.min_value,
            "max_value": payload.max_value,
        }
    gid = upsert_gauge(DB_PATH, payload.name, payload.camera_id, roi=roi, calibration=cal, gauge_id=payload.gauge_id)
    return {"ok": True, "gauge_id": gid}


@router.post("/gauges/{gauge_id}/calibration")
def set_calibration(gauge_id: int, payload: CalibrationIn):
    from main import DB_PATH
    init_models(DB_PATH)
    set_gauge_calibration(DB_PATH, gauge_id, payload.min_angle, payload.max_angle, payload.min_value, payload.max_value)
    return {"ok": True}


@router.post("/gauges/{gauge_id}/thresholds")
def set_thresholds(gauge_id: int, payload: ThresholdsIn):
    from main import DB_PATH
    init_models(DB_PATH)
    set_gauge_thresholds(DB_PATH, gauge_id, payload.low_warn, payload.high_warn, payload.low_crit, payload.high_crit,
                         payload.roc_limit, payload.cusum_k, payload.cusum_h)
    return {"ok": True}


@router.get("/gauges/{gauge_id}/readings")
def get_readings(gauge_id: int, since: Optional[str] = None):
    from main import DB_PATH
    init_models(DB_PATH)
    rows = recent_gauge_readings(DB_PATH, gauge_id, since)
    return {"gauge_id": gauge_id, "readings": rows}


@router.get("/events")
def get_events(severity: Optional[str] = None, limit: int = 200):
    from main import DB_PATH
    init_models(DB_PATH)
    rows = recent_events(DB_PATH, severity, limit)
    return {"events": rows}

