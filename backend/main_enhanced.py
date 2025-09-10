# backend/main_enhanced.py
"""
Enhanced Industrial Monitoring API with ML Integration
Supports YOLO/VDN models, multi-camera grid, interactive calibration
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
import cv2
import numpy as np
import sqlite3
from datetime import datetime, timedelta
import asyncio
import base64
import io
import json
from typing import List, Optional, Dict, Any
from pathlib import Path
import logging

# Import existing modules
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

# Import new ML detector
from ml_dial_cv import create_enhanced_detector, ModelManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DB_PATH = "monitoring.db"
app = FastAPI(title="Industrial Monitoring API ML-Enhanced", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============= Configuration Models =============
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


class CalibrationClick(BaseModel):
    camera_id: str
    clicks: List[List[int]]  # [[x1, y1], [x2, y2]]
    image_b64: str


class MLConfig(BaseModel):
    backend: str = 'hybrid'  # 'yolo', 'vdn', 'classical', 'hybrid'
    enable_tracking: bool = True
    confidence_threshold: float = 0.5


class FewShotSample(BaseModel):
    camera_id: str
    image_b64: str
    true_reading: float


class GridLayoutConfig(BaseModel):
    layout: str = '2x2'  # '2x2', '3x3', '4x4', 'auto'
    camera_ids: List[str]


# ============= Global State =============
# Per-camera detector instances
camera_detectors: Dict[str, Any] = {}
model_manager = ModelManager()

# Global fallback thresholds
GLOBAL_THRESHOLD_LOW = 85.0
GLOBAL_THRESHOLD_MED = 90.0
GLOBAL_THRESHOLD_HIGH = 95.0

# WebSocket connections
active_ws: List[WebSocket] = []
camera_ws: Dict[str, List[WebSocket]] = {}  # Per-camera subscriptions

# Grid layout configuration
grid_layout = {
    'layout': '2x2',
    'cameras': []
}

# ML configuration
ml_config = MLConfig()


# ============= Helper Functions =============
def get_or_create_detector(camera_id: str):
    """Get or create ML detector for camera"""
    if camera_id not in camera_detectors:
        camera_detectors[camera_id] = create_enhanced_detector(
            backend=ml_config.backend,
            enable_tracking=ml_config.enable_tracking
        )
        logger.info(f"Created {ml_config.backend} detector for camera {camera_id}")
    return camera_detectors[camera_id]


def decode_base64_image(image_b64: str) -> np.ndarray:
    """Decode base64 image to numpy array"""
    img_data = base64.b64decode(image_b64)
    nparr = np.frombuffer(img_data, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)


def encode_image_base64(image: np.ndarray) -> str:
    """Encode numpy array to base64"""
    _, buffer = cv2.imencode('.jpg', image)
    return base64.b64encode(buffer).decode('utf-8')


# ============= Startup/Shutdown =============
@app.on_event("startup")
async def startup():
    """Initialize database and download models"""
    init_database(DB_PATH)

    # Pre-download models in background
    asyncio.create_task(download_models())

    logger.info("Enhanced Industrial Monitoring API started")


async def download_models():
    """Download ML models in background"""
    try:
        logger.info("Downloading ML models...")
        model_manager.download_model('yolov5s_gauge')
        model_manager.download_model('vdn_resnet34')
        logger.info("Model download complete")
    except Exception as e:
        logger.error(f"Model download failed: {e}")


# ============= WebSocket Endpoints =============
@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    """Main WebSocket for dashboard"""
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
                "grid_layout": grid_layout,
                "ml_config": ml_config.dict(),
                "threshold": GLOBAL_THRESHOLD_LOW,
            })
    except WebSocketDisconnect:
        pass
    finally:
        if ws in active_ws:
            active_ws.remove(ws)


@app.websocket("/ws/camera/{camera_id}")
async def ws_camera_endpoint(ws: WebSocket, camera_id: str):
    """Per-camera WebSocket for real-time updates"""
    await ws.accept()

    if camera_id not in camera_ws:
        camera_ws[camera_id] = []
    camera_ws[camera_id].append(ws)

    try:
        while True:
            await asyncio.sleep(0.1)  # Higher frequency for camera streams
    except WebSocketDisconnect:
        pass
    finally:
        if camera_id in camera_ws and ws in camera_ws[camera_id]:
            camera_ws[camera_id].remove(ws)


# ============= ML Configuration Endpoints =============
@app.get("/api/ml/config")
async def get_ml_config():
    """Get current ML configuration"""
    return ml_config.dict()


@app.post("/api/ml/config")
async def set_ml_config(config: MLConfig):
    """Update ML configuration"""
    global ml_config
    ml_config = config

    # Clear existing detectors to force recreation with new config
    camera_detectors.clear()

    return {"ok": True, "config": ml_config.dict()}


@app.get("/api/ml/models")
async def get_model_status():
    """Get status of ML models"""
    models = {
        'yolov5s_gauge': (model_manager.cache_dir / 'yolov5s_gauge.pt').exists(),
        'vdn_resnet34': (model_manager.cache_dir / 'vdn_resnet34.pt').exists(),
    }
    return {"models": models}


@app.post("/api/ml/download")
async def trigger_model_download():
    """Manually trigger model download"""
    asyncio.create_task(download_models())
    return {"ok": True, "message": "Download started"}


# ============= Enhanced Processing Endpoint =============
@app.post("/api/process_frame_ml")
async def process_frame_ml(camera_id: str, file: UploadFile = File(...)):
    """Process frame with ML detector"""
    data = await file.read()
    arr = np.frombuffer(data, np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    if frame is None:
        return JSONResponse(
            status_code=400,
            content={"error": "Invalid image"}
        )

    # Get detector for this camera
    detector = get_or_create_detector(camera_id)

    # Detect dial
    result = detector.detect_dial(frame)

    # Get thresholds
    low_t, med_t, high_t = get_camera_thresholds(
        DB_PATH, camera_id,
        default_low=GLOBAL_THRESHOLD_LOW,
        default_med=GLOBAL_THRESHOLD_MED,
        default_high=GLOBAL_THRESHOLD_HIGH
    )

    # Determine status
    detected = result.get('detected', False)
    reading = result.get('reading', 0.0) if detected else 0.0

    status = "UNKNOWN"
    is_anom = False

    if detected:
        if reading > high_t:
            status = "HIGH"
            is_anom = True
        elif reading > med_t:
            status = "MEDIUM"
            is_anom = True
        elif reading > low_t:
            status = "LOW"
            is_anom = True
        else:
            status = "NORMAL"

    # Store in database
    insert_reading(
        DB_PATH, camera_id, reading,
        int(is_anom), result.get('confidence', 0.0), int(detected)
    )
    upsert_camera_status(DB_PATH, camera_id, reading, is_active=1)

    if is_anom:
        insert_anomaly(DB_PATH, camera_id, reading, low_t, status)

    # Annotate image
    annotated = frame.copy()
    if detected and 'center' in result:
        cx, cy = int(result['center'][0]), int(result['center'][1])
        radius = int(result.get('radius', 100))

        # Draw circle
        cv2.circle(annotated, (cx, cy), radius, (0, 255, 0), 2)

        # Draw pointer
        if 'angle' in result:
            angle_rad = np.radians(result['angle'])
            px = int(cx + radius * 0.8 * np.cos(angle_rad))
            py = int(cy + radius * 0.8 * np.sin(angle_rad))
            cv2.line(annotated, (cx, cy), (px, py), (255, 0, 0), 3)

        # Add text
        cv2.putText(annotated, f"{reading:.1f}", (cx - 30, cy - radius - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(annotated, f"Method: {result.get('method', 'unknown')}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)

    # Prepare response
    response = {
        "detected": detected,
        "status": status,
        "reading": reading,
        "confidence": result.get('confidence', 0.0),
        "method": result.get('method', 'unknown'),
        "angle": result.get('angle', 0.0),
        "annotated_image_b64": encode_image_base64(annotated),
        "is_anomaly": is_anom,
        "thresholds": {"low": low_t, "med": med_t, "high": high_t},
        "smoothed": result.get('smoothed', False)
    }

    # Broadcast to WebSocket subscribers
    await broadcast_camera_update(camera_id, response)

    return response


async def broadcast_camera_update(camera_id: str, data: Dict):
    """Broadcast update to camera subscribers"""
    if camera_id in camera_ws:
        for ws in list(camera_ws[camera_id]):
            try:
                await ws.send_json({
                    "type": "camera_update",
                    "camera_id": camera_id,
                    **data
                })
            except:
                pass

    # Also broadcast to main dashboard
    for ws in list(active_ws):
        try:
            await ws.send_json({
                "type": "reading_update",
                "camera_id": camera_id,
                **data
            })
        except:
            pass


# ============= Calibration Endpoints =============
@app.post("/api/calibrate/interactive")
async def calibrate_interactive(data: CalibrationClick):
    """Interactive two-click calibration"""
    detector = get_or_create_detector(data.camera_id)

    # Decode image
    image = decode_base64_image(data.image_b64)

    # Convert clicks to tuples
    clicks = [(click[0], click[1]) for click in data.clicks]

    # Perform calibration
    success = detector.calibrate_interactive(image, clicks)

    if not success:
        return JSONResponse(
            status_code=400,
            content={"error": "Calibration failed - need 2 clicks and detected dial"}
        )

    return {
        "ok": True,
        "camera_id": data.camera_id,
        "min_angle": detector.min_angle,
        "max_angle": detector.max_angle
    }


@app.post("/api/calibrate/values/{camera_id}")
async def set_value_range(camera_id: str, min_value: float = 0, max_value: float = 100):
    """Set min/max values for gauge"""
    detector = get_or_create_detector(camera_id)
    detector.set_value_range(min_value, max_value)

    return {
        "ok": True,
        "camera_id": camera_id,
        "min_value": min_value,
        "max_value": max_value
    }


# ============= Few-Shot Learning Endpoints =============
@app.post("/api/ml/few_shot")
async def few_shot_training(samples: List[FewShotSample]):
    """Few-shot adaptation with labeled samples"""
    results = {}

    for sample in samples:
        detector = get_or_create_detector(sample.camera_id)
        image = decode_base64_image(sample.image_b64)

        # Add to training samples
        if not hasattr(detector, 'training_samples'):
            detector.training_samples = []

        detector.training_samples.append((image, sample.true_reading))

        # Perform adaptation every 5 samples
        if len(detector.training_samples) >= 5:
            detector.few_shot_adapt(detector.training_samples[-5:])
            results[sample.camera_id] = "adapted"
        else:
            results[sample.camera_id] = f"collected ({len(detector.training_samples)}/5)"

    return {"ok": True, "results": results}


# ============= Grid Layout Endpoints =============
@app.get("/api/grid/layout")
async def get_grid_layout():
    """Get current grid layout"""
    return grid_layout


@app.post("/api/grid/layout")
async def set_grid_layout(config: GridLayoutConfig):
    """Set grid layout configuration"""
    global grid_layout
    grid_layout = {
        'layout': config.layout,
        'cameras': config.camera_ids
    }
    return {"ok": True, "layout": grid_layout}


@app.get("/api/grid/stream")
async def get_grid_stream():
    """Get combined grid stream data"""
    camera_data = {}

    for camera_id in grid_layout['cameras']:
        # Get latest reading
        readings = get_readings(DB_PATH, camera_id, datetime.now() - timedelta(minutes=1))
        camera_data[camera_id] = {
            'camera_id': camera_id,
            'latest_reading': readings[0] if readings else None,
            'status': 'active' if readings else 'inactive'
        }

    return {
        'layout': grid_layout['layout'],
        'cameras': camera_data
    }


# ============= Batch Processing =============
@app.post("/api/batch/process")
async def batch_process(camera_id: str, files: List[UploadFile] = File(...)):
    """Process multiple frames in batch"""
    detector = get_or_create_detector(camera_id)
    results = []

    for file in files:
        data = await file.read()
        arr = np.frombuffer(data, np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)

        if frame is not None:
            result = detector.detect_dial(frame)
            results.append({
                'filename': file.filename,
                'detected': result.get('detected', False),
                'reading': result.get('reading', 0.0),
                'confidence': result.get('confidence', 0.0)
            })

    return {"camera_id": camera_id, "results": results}


# ============= Analytics Endpoints =============
@app.get("/api/analytics/trends/{camera_id}")
async def get_trends(camera_id: str, hours: int = 24):
    """Get trend analysis for camera"""
    readings = get_readings(DB_PATH, camera_id,
                            datetime.now() - timedelta(hours=hours))

    if not readings:
        return {"error": "No data available"}

    values = [r['reading'] for r in readings]

    analytics = {
        'camera_id': camera_id,
        'period_hours': hours,
        'total_readings': len(readings),
        'mean': np.mean(values),
        'std': np.std(values),
        'min': np.min(values),
        'max': np.max(values),
        'trend': 'stable'  # Calculate trend
    }

    # Simple trend detection
    if len(values) > 10:
        first_half = np.mean(values[:len(values) // 2])
        second_half = np.mean(values[len(values) // 2:])

        if second_half > first_half * 1.1:
            analytics['trend'] = 'increasing'
        elif second_half < first_half * 0.9:
            analytics['trend'] = 'decreasing'

    return analytics


@app.get("/api/analytics/accuracy")
async def get_accuracy_metrics():
    """Get accuracy metrics for ML models"""
    metrics = {
        'cameras': {}
    }

    for camera_id, detector in camera_detectors.items():
        if hasattr(detector, 'tracking_history'):
            history = detector.tracking_history
            if len(history) > 1:
                # Calculate stability (inverse of variance)
                stability = 1.0 / (1.0 + np.var(history))
                metrics['cameras'][camera_id] = {
                    'stability': stability,
                    'samples': len(history),
                    'backend': detector.backend
                }

    return metrics


# ============= Legacy Endpoints (Backward Compatibility) =============
@app.post("/api/process_frame")
async def process_frame_legacy(camera_id: str, file: UploadFile = File(...)):
    """Legacy endpoint - redirects to ML version"""
    return await process_frame_ml(camera_id, file)


@app.get("/api/threshold")
async def get_threshold():
    return {"threshold": GLOBAL_THRESHOLD_LOW}


@app.post("/api/threshold")
async def set_threshold(payload: ThresholdIn):
    global GLOBAL_THRESHOLD_LOW
    GLOBAL_THRESHOLD_LOW = float(payload.threshold)
    return {"ok": True, "threshold": GLOBAL_THRESHOLD_LOW}


@app.post("/api/calibrate")
async def set_calibration_legacy(payload: CalibrateIn):
    """Legacy calibration - applies to all cameras"""
    for detector in camera_detectors.values():
        detector.min_angle = payload.min_angle
        detector.max_angle = payload.max_angle
        detector.min_value = payload.min_value
        detector.max_value = payload.max_value

    return {"ok": True, "calibration": payload.dict()}


# Keep all other existing endpoints...
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

    import pandas as pd
    df = pd.read_sql_query(q, conn, params=params)
    conn.close()

    sio = io.StringIO()
    df.to_csv(sio, index=False)
    sio.seek(0)

    return StreamingResponse(
        io.BytesIO(sio.getvalue().encode()),
        media_type="text/csv",
        headers={
            "Content-Disposition": f"attachment; filename=export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        }
    )


@app.get("/api/report/pdf")
async def pdf(date: Optional[str] = None):
    if not date:
        date = datetime.now().strftime('%Y-%m-%d')
    pdf_bytes = build_daily_report_pdf(DB_PATH, date)

    return StreamingResponse(
        io.BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers={
            "Content-Disposition": f"attachment; filename=report_{date}.pdf"
        }
    )