
import asyncio
import base64
import io
import json
from datetime import datetime, timedelta
from typing import List, Optional, Dict

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from dial_cv import DialReader
from database import (
    init_database, insert_reading, insert_anomaly, upsert_camera_status,
    recent_anomalies, camera_status_all, get_readings, calc_metrics,
    get_camera_thresholds, set_camera_thresholds
)
from report import build_daily_report_pdf

DB_PATH = "monitoring.db"
app = FastAPI(title="Industrial Monitor Pro - Unified ML API", version="3.1.5") # Version bump
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

class AppState:
    detector: Optional[DialReader] = None
    active_ws: List[WebSocket] = []

state = AppState()

class CamThresholdsIn(BaseModel): low: float; med: float; high: float
class CalibrateIn(BaseModel): min_angle: float = 0; max_angle: float = 270; min_value: float = 0; max_value: float = 100

@app.on_event("startup")
async def startup_event():
    print("Server starting up...")
    init_database(DB_PATH)
    try:
        state.detector = DialReader(config_path="config.json")
        print("ML Dial Reader loaded successfully.")
    except FileNotFoundError as e:
        print(f"ERROR: Could not initialize Dial Reader: {e}")
        state.detector = None

@app.post("/api/process_frame")
async def process_frame(camera_id: str, file: UploadFile = File(...)):
    if not state.detector:
        raise HTTPException(status_code=503, detail="ML Detector not initialized.")
    
    data = await file.read()
    arr = np.frombuffer(data, np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        return {"detected": False, "reading": 0.0}

    ml_readings = state.detector.detect(frame)
    detected = bool(ml_readings)
    reading = ml_readings[0]['value'] if detected else 0.0
    confidence = ml_readings[0]['confidence'] if detected else 0.0
    
    gauge_bbox = None
    pointer_bbox = None

    if detected:
        low_t, med_t, high_t = get_camera_thresholds(DB_PATH, camera_id, default_low=10, default_med=20, default_high=30)
        status = "UNKNOWN"
        is_anom = False
        band_threshold_used = None

        if reading > high_t:
            status, is_anom, band_threshold_used = "HIGH", True, high_t
        elif reading > med_t:
            status, is_anom, band_threshold_used = "MEDIUM", True, med_t
        elif reading > low_t:
            status, is_anom, band_threshold_used = "LOW", True, low_t
        else:
            status, is_anom, band_threshold_used = "NORMAL", False, low_t
        
        insert_reading(DB_PATH, camera_id, reading, int(is_anom), float(confidence), int(detected))
        upsert_camera_status(DB_PATH, camera_id, reading, is_active=1)
        if is_anom:
            insert_anomaly(DB_PATH, camera_id, reading, float(band_threshold_used or low_t), status)

        main_reading = ml_readings[0]
        if 'bounds' in main_reading:
            b = main_reading['bounds']
            gauge_bbox = [b['x'], b['y'], b['x'] + b['w'], b['y'] + b['h']]
        
        if 'pointer_bounds' in main_reading:
            pb = main_reading['pointer_bounds']
            pointer_bbox = [pb['x'], pb['y'], pb['x'] + pb['w'], pb['y'] + pb['h']]
    else:
        status = "UNKNOWN"
        is_anom = False
        low_t, med_t, high_t = get_camera_thresholds(DB_PATH, camera_id, default_low=10, default_med=20, default_high=30)

    payload = {
        "detected": detected,
        "status": status,
        "reading": reading,
        "confidence": float(confidence),
        "is_anomaly": is_anom,
        "thresholds": {"low": low_t, "med": med_t, "high": high_t},
        "gauge_bbox": gauge_bbox,
        "pointer_bbox": pointer_bbox,
    }
    
    for ws in list(state.active_ws):
        try:
            await ws.send_json({"type": "reading_update", "camera_id": camera_id, **payload})
        except Exception:
            pass
    
    return payload

# --- Other Endpoints (Unchanged) ---

@app.get("/api/config")
async def get_config():
    if not state.detector: raise HTTPException(status_code=503, detail="Detector not initialized")
    return state.detector.config

@app.post("/api/config")
async def set_config(new_config: Dict):
    config_path = "config.json"
    try:
        with open(config_path, 'w') as f: json.dump(new_config, f, indent=2)
        if state.detector: state.detector.config = new_config
        return {"message": "Config saved. Restart may be needed."}
    except Exception as e: raise HTTPException(status_code=500, detail=f"Failed to write config: {e}")

@app.post("/api/camera/{camera_id}/thresholds")
async def api_set_cam_thresholds(camera_id: str, payload: CamThresholdsIn):
    set_camera_thresholds(DB_PATH, camera_id, payload.low, payload.med, payload.high)
    return {"ok": True, "camera_id": camera_id, **payload.dict()}

@app.get("/api/camera/{camera_id}/thresholds")
async def api_get_cam_thresholds(camera_id: str):
    low, med, high = get_camera_thresholds(DB_PATH, camera_id, default_low=10, default_med=20, default_high=30)
    return {"camera_id": camera_id, "low": low, "med": med, "high": high}

@app.get("/api/readings/{camera_id}")
async def readings(camera_id: str, hours: int = 24):
    since = datetime.now() - timedelta(hours=hours)
    rows = get_readings(DB_PATH, camera_id, since)
    return {"camera_id": camera_id, "readings": rows}

@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    state.active_ws.append(ws)
    try:
        while True:
            await asyncio.sleep(1)
            await ws.send_json({
                "timestamp": datetime.now().isoformat(), "metrics": calc_metrics(DB_PATH),
                "anomalies": recent_anomalies(DB_PATH, limit=8), "cameras": camera_status_all(DB_PATH),
            })
    except WebSocketDisconnect: pass
    finally: 
        if ws in state.active_ws: state.active_ws.remove(ws)

@app.get("/api/report/pdf")
async def pdf(date: Optional[str] = None):
    date_str = date or datetime.now().strftime('%Y-%m-%d')
    pdf_bytes = build_daily_report_pdf(DB_PATH, date_str)
    return StreamingResponse(io.BytesIO(pdf_bytes), media_type="application/pdf", headers={"Content-Disposition": f"attachment; filename=report_{date_str}.pdf"})

if __name__ == "__main__":
    print("--- Starting Industrial Monitor Unified ML API ---")
    uvicorn.run(app, host="0.0.0.0", port=8000)
