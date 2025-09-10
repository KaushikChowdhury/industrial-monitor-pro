import os
import io
import numpy as np
import cv2
from fastapi.testclient import TestClient

import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from main import app  # noqa

client = TestClient(app)


def test_threshold_get_set():
    r = client.get('/api/threshold'); assert r.status_code == 200
    old = r.json()['threshold']
    r = client.post('/api/threshold', json={'threshold': 77}); assert r.status_code == 200
    assert r.json()['threshold'] == 77
    client.post('/api/threshold', json={'threshold': old})


def test_report_pdf_generation():
    r = client.get('/api/report/pdf'); assert r.status_code == 200
    assert r.headers['content-type'] == 'application/pdf'


def test_process_frame_with_blank_image_unknown_status():
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    ok, buf = cv2.imencode('.jpg', img); assert ok
    files = {'file': ('frame.jpg', io.BytesIO(buf.tobytes()), 'image/jpeg')}
    r = client.post('/api/process_frame?camera_id=cam_test', files=files)
    assert r.status_code == 200
    js = r.json()
    assert js['reading'] == 0.0
    assert js['status'] == 'UNKNOWN'


def test_per_camera_thresholds_crud():
    cam = 'cam_X'
    r = client.post(f'/api/camera/{cam}/thresholds', json={'low': 60, 'med': 80, 'high': 90})
    assert r.status_code == 200
    r = client.get(f'/api/camera/{cam}/thresholds')
    assert r.status_code == 200
    data = r.json()
    assert data['low'] == 60 and data['med'] == 80 and data['high'] == 90
