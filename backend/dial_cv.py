
import cv2
import numpy as np
import torch
import json
from pathlib import Path
from dataclasses import dataclass

@dataclass
class _ClassicNeedleFinder:
    min_angle: float = 0.0
    max_angle: float = 270.0
    min_value: float = 0.0
    max_value: float = 100.0

    def find_needle(self, dial_image):
        try:
            gray = cv2.cvtColor(dial_image, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape
            cx, cy, r = w // 2, h // 2, min(w, h) // 2

            mask = np.zeros_like(gray)
            cv2.circle(mask, (cx, cy), r, 255, -1)
            inner_r = max(5, int(r * 0.2))
            cv2.circle(mask, (cx, cy), inner_r, 0, -1)
            dial_roi = cv2.bitwise_and(gray, mask)

            v = np.median(dial_roi[dial_roi > 0]) if np.any(dial_roi > 0) else 0
            edges = cv2.Canny(dial_roi, int(max(0, 0.66 * v)), int(min(255, 1.33 * v)))

            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=30, minLineLength=int(r*0.4), maxLineGap=10)

            angle = None
            best_len = 0
            if lines is not None:
                for l in lines:
                    x1, y1, x2, y2 = l[0]
                    d1 = ((x1-cx)**2 + (y1-cy)**2)**0.5
                    d2 = ((x2-cx)**2 + (y2-cy)**2)**0.5
                    if min(d1, d2) < r * 0.3 and max(d1, d2) > r * 0.5:
                        length = abs(d1 - d2)
                        if length > best_len:
                            best_len = length
                            fx, fy = (x1, y1) if d1 > d2 else (x2, y2)
                            angle = (np.degrees(np.arctan2(fy - cy, fx - cx)) + 360) % 360
            
            if angle is None: return {"detected": False}

            sweep = (angle - self.min_angle) % 360
            total_span = (self.max_angle - self.min_angle) % 360
            norm = np.clip(sweep / total_span, 0, 1) if total_span > 0 else 0
            value = self.min_value + norm * (self.max_value - self.min_value)

            return {"detected": True, "angle": float(angle), "reading": float(value), "confidence": 0.9}
        except Exception:
            return {"detected": False}

class DialReader:
    def __init__(self, config_path="config.json"):
        print("Initializing Dial Reader...")
        self.config = self._load_config(config_path)
        self.device = self._get_device()
        self.model = self._load_model()
        self.needle_finder = _ClassicNeedleFinder()
        print("Dial Reader initialized.")

    def _load_config(self, config_path):
        if not Path(config_path).exists():
            raise FileNotFoundError(f"Config file '{config_path}' not found. Run setup_ml.py --config")
        with open(config_path, 'r') as f: return json.load(f)

    def _get_device(self):
        req_device = self.config.get("device", "auto").lower()
        if req_device == "auto": return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(req_device)

    def _load_model(self):
        model_path = self.config.get("model_path", "models/yolov5s.pt")
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model file '{model_path}' not found. Run setup_ml.py --download yolov5s")
        
        print(f"Loading model from '{model_path}' onto {self.device}...")
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, _verbose=False)
        model.to(self.device)
        # FIX: Lowered confidence and updated IOU for better detection chances
        model.conf = self.config.get("confidence_threshold", 0.25)
        model.iou = self.config.get("iou_threshold", 0.45)
        print(f"Model loaded. Confidence threshold: {model.conf}")
        return model

    def update_calibration(self, min_angle, max_angle, min_value, max_value):
        self.needle_finder.min_angle = min_angle
        self.needle_finder.max_angle = max_angle
        self.needle_finder.min_value = min_value
        self.needle_finder.max_value = max_value

    def detect(self, image):
        results = self.model(image)
        predictions = results.pandas().xyxy[0]

        # --- DEBUGGING BLOCK ---
        if len(predictions) > 0:
            print("[DEBUG] Raw YOLO Predictions:")
            print(predictions[['name', 'confidence']])
        # --- END DEBUGGING ---

        # FIX: Accept both 'dial' (custom) and 'clock' (standard YOLO) as valid classes.
        dial_preds = predictions[predictions['name'].isin(['dial', 'clock'])]

        readings = []
        for i, pred in dial_preds.iterrows():
            x1, y1, x2, y2 = int(pred['xmin']), int(pred['ymin']), int(pred['xmax']), int(pred['ymax'])
            dial_crop = image[y1:y2, x1:x2]
            if dial_crop.size == 0: continue

            needle_result = self.needle_finder.find_needle(dial_crop)

            if needle_result["detected"]:
                readings.append({
                    "dialId": f"dial_{i}",
                    "value": needle_result["reading"],
                    "confidence": float(pred['confidence']) * needle_result["confidence"],
                    "angle": needle_result.get("angle"),
                    "bounds": {"x": x1, "y": y1, "w": x2 - x1, "h": y2 - y1}
                })
        
        return readings
