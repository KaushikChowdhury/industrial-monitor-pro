from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import cv2
import numpy as np
import sqlite3
from datetime import datetime, timedelta
import asyncio
import base64
import io
from typing import List, Optional

from dial_cv import DialDetector
from database import (
    init_database,
    insert_reading,
    insert_anomaly,
    upsert_camera_status,
    recent_anomalies,
    camera_status_all,
    get_readings,
    calc_metrics,
    get_camera_thresholds,
    set_camera_thresholds,
)
from report import build_daily_report_pdf

DB_PATH = "monitoring.db"
app = FastAPI(title="Industrial Monitoring API", version="1.1.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

detector = DialDetector()
# Global fallback (camera-level overrides)
GLOBAL_THRESHOLD_LOW = 85.0
GLOBAL_THRESHOLD_MED = 90.0
GLOBAL_THRESHOLD_HIGH = 95.0
active_ws: List[WebSocket] = []

class ThresholdIn(BaseModel):
    threshold: float

class CamThresholdsIn(BaseModel):
    low: float
    med: float
    high: float

class CalibrateIn(BaseModel):
    min_angle: float = 0
    max_angle: float = 270
    min_value: float = 0
    max_value: float = 100

@app.on_event("startup")
async def _startup():
    init_database(DB_PATH)

@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    active_ws.append(ws)
    try:
        while True:
            await asyncio.sleep(1)
            metrics = calc_metrics(DB_PATH)
            anomalies = recent_anomalies(DB_PATH, limit=8)
            cameras = camera_status_all(DB_PATH)
            await ws.send_json({
                "timestamp": datetime.now().isoformat(),
                "metrics": metrics,
                "anomalies": anomalies,
                "cameras": cameras,
                # keep global fallback for legacy UI slider
                "threshold": GLOBAL_THRESHOLD_LOW,
            })
    except WebSocketDisconnect:
        pass
    finally:
        if ws in active_ws:
            active_ws.remove(ws)

# Legacy global threshold (fallback)
@app.get("/api/threshold")
async def get_threshold():
    return {"threshold": GLOBAL_THRESHOLD_LOW}

@app.post("/api/threshold")
async def set_threshold(payload: ThresholdIn):
    global GLOBAL_THRESHOLD_LOW
    GLOBAL_THRESHOLD_LOW = float(payload.threshold)
    return {"ok": True, "threshold": GLOBAL_THRESHOLD_LOW}

# Per-camera thresholds
@app.get("/api/camera/{camera_id}/thresholds")
async def api_get_cam_thresholds(camera_id: str):
    low, med, high = get_camera_thresholds(DB_PATH, camera_id,
                                           default_low=GLOBAL_THRESHOLD_LOW,
                                           default_med=GLOBAL_THRESHOLD_MED,
                                           default_high=GLOBAL_THRESHOLD_HIGH)
    return {"camera_id": camera_id, "low": low, "med": med, "high": high}

@app.post("/api/camera/{camera_id}/thresholds")
async def api_set_cam_thresholds(camera_id: str, payload: CamThresholdsIn):
    set_camera_thresholds(DB_PATH, camera_id, payload.low, payload.med, payload.high)
    return {"ok": True, "camera_id": camera_id, **payload.dict()}

@app.post("/api/calibrate")
async def set_calibration(payload: CalibrateIn):
    detector.min_angle = payload.min_angle
    detector.max_angle = payload.max_angle
    detector.min_value = payload.min_value
    detector.max_value = payload.max_value
    return {"ok": True, "calibration": payload.dict()}

@app.post("/api/process_frame")
async def process_frame(camera_id: str, file: UploadFile = File(...)):
    data = await file.read()
    arr = np.frombuffer(data, np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    # Guard: decoding may fail on corrupt uploads
    if frame is None:
        payload = {
            "detected": False,
            "status": "UNKNOWN",
            "reading": 0.0,
            "confidence": 0.0,
            "annotated_image_b64": "",
            "is_anomaly": False,
            "thresholds": {"low": GLOBAL_THRESHOLD_LOW, "med": GLOBAL_THRESHOLD_MED, "high": GLOBAL_THRESHOLD_HIGH},
        }
        insert_reading(DB_PATH, camera_id, 0.0, 0, 0.0, 0)
        upsert_camera_status(DB_PATH, camera_id, 0.0, is_active=1)
        return payload

    result = detector.detect_dial(frame)

    # Determine thresholds for this camera
    low_t, med_t, high_t = get_camera_thresholds(DB_PATH, camera_id,
        default_low=GLOBAL_THRESHOLD_LOW, default_med=GLOBAL_THRESHOLD_MED, default_high=GLOBAL_THRESHOLD_HIGH)

    detected = bool(result.get("detected", False))
    reading = float(result.get("reading", 0.0)) if detected else 0.0

    # Status band logic
    status = "UNKNOWN"
    is_anom = False
    band_threshold_used: Optional[float] = None
    if detected:
        if reading > high_t:
            status = "HIGH"; is_anom = True; band_threshold_used = high_t
        elif reading > med_t:
            status = "MEDIUM"; is_anom = True; band_threshold_used = med_t
        elif reading > low_t:
            status = "LOW"; is_anom = True; band_threshold_used = low_t
        else:
            status = "NORMAL"; is_anom = False; band_threshold_used = low_t

    insert_reading(DB_PATH, camera_id, reading, int(is_anom), float(result.get("confidence", 0.0)), int(detected))
    upsert_camera_status(DB_PATH, camera_id, reading, is_active=1)
    if is_anom:
        insert_anomaly(DB_PATH, camera_id, reading, float(band_threshold_used or low_t), status)

    # Annotate (optional overlay consumers will use this)
    annotated = frame.copy()
    if detected and result.get("center"):
        cx, cy = result["center"]
        radius = int(result["radius"])
        cv2.circle(annotated, (int(cx), int(cy)), radius, (0, 255, 0), 2)
        cv2.putText(
            annotated,
            f"{reading:.1f}", (int(cx) - 10, int(cy) - radius - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
        )
    _, buf = cv2.imencode('.jpg', annotated)
    b64 = base64.b64encode(buf.tobytes()).decode('utf-8')

    payload = {
        "detected": detected,
        "status": status,  # UNKNOWN / NORMAL / LOW / MEDIUM / HIGH
        "reading": reading,
        "confidence": float(result.get("confidence", 0.0)),
        "annotated_image_b64": b64,
        "is_anomaly": is_anom,
        "thresholds": {"low": low_t, "med": med_t, "high": high_t},
    }

    # Best-effort push to websockets
    for ws in list(active_ws):
        try:
            await ws.send_json({"type": "reading_update", "camera_id": camera_id, **payload})
        except Exception:
            pass

    return payload

@app.get("/api/readings/{camera_id}")
async def readings(camera_id: str, hours: int = 24):
    since = datetime.now() - timedelta(hours=hours)
    rows = get_readings(DB_PATH, camera_id, since)
    return {"camera_id": camera_id, "readings": rows}

@app.get("/api/anomalies")
async def anomalies():
    return {"anomalies": recent_anomalies(DB_PATH, 100)}

@app.get("/api/cameras")
async def cameras():
    return {"cameras": camera_status_all(DB_PATH)}

@app.get("/api/metrics")
async def metrics():
    return calc_metrics(DB_PATH)

@app.get("/api/export/csv")
async def export_csv(start: Optional[str] = None, end: Optional[str] = None):
    conn = sqlite3.connect(DB_PATH)
    q = "SELECT * FROM dial_readings"
    params = []
    if start and end:
        q += " WHERE timestamp BETWEEN ? AND ?"
        params = [start, end]
    df = __import__("pandas").read_sql_query(q, conn, params=params)
    conn.close()
    sio = io.StringIO(); df.to_csv(sio, index=False); sio.seek(0)
    return StreamingResponse(io.BytesIO(sio.getvalue().encode()), media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"})

@app.get("/api/report/pdf")
async def pdf(date: Optional[str] = None):
    if not date:
        date = datetime.now().strftime('%Y-%m-%d')
    pdf_bytes = build_daily_report_pdf(DB_PATH, date)
    return StreamingResponse(io.BytesIO(pdf_bytes), media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename=report_{date}.pdf"})