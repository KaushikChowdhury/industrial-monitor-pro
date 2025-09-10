
import cv2
import numpy as np
import torch
import json
from pathlib import Path
from dataclasses import dataclass
import torchvision.transforms as transforms
from PIL import Image

# (The _ClassicNeedleFinder class remains unchanged as a fallback)
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

            return {"detected": True, "angle": float(angle), "reading": float(value), "confidence": 0.7}
        except Exception:
            return {"detected": False}

class MLNeedleFinder:
    """Finds a needle in a cropped dial image using a Vector Detection Network (VDN)."""
    def __init__(self, config, device):
        self.config = config
        self.device = device
        self.model = self._load_vdn_model()
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.min_angle, self.max_angle, self.min_value, self.max_value = 0, 270, 0, 100

    def _load_vdn_model(self):
        model_config = self.config.get("models", {})
        vdn_model_path = model_config.get("vdn_model_path", "models/vdn_resnet34.pt")
        if not Path(vdn_model_path).exists():
            print(f"WARNING: VDN model file '{vdn_model_path}' not found. ML finder will be disabled.")
            return None
        try:
            print(f"Loading VDN model from '{vdn_model_path}' onto {self.device}...")
            model = torch.load(vdn_model_path, map_location=self.device)
            model.eval()
            print("VDN model loaded successfully.")
            return model
        except Exception as e:
            print(f"ERROR loading VDN model: {e}. ML finder will be disabled.")
            return None

    def update_calibration(self, min_angle, max_angle, min_value, max_value):
        self.min_angle, self.max_angle, self.min_value, self.max_value = min_angle, max_angle, min_value, max_value

    def find_needle(self, dial_image):
        if self.model is None: return {"detected": False, "reason": "VDN model not loaded."}
        try:
            img_pil = Image.fromarray(cv2.cvtColor(dial_image, cv2.COLOR_BGR2RGB))
            img_tensor = self.transform(img_pil).unsqueeze(0).to(self.device)

            with torch.no_grad():
                output = self.model(img_tensor)

            vec = output.squeeze().cpu().numpy()
            angle = (np.degrees(np.arctan2(vec[1], vec[0])) + 360) % 360

            sweep = (angle - self.min_angle) % 360
            total_span = (self.max_angle - self.min_angle) % 360
            if total_span <= 0: total_span = 360
            
            norm = np.clip(sweep / total_span, 0, 1)
            value = self.min_value + norm * (self.max_value - self.min_value)

            return {"detected": True, "angle": float(angle), "reading": float(value), "confidence": 0.95}
        except Exception as e:
            return {"detected": False, "reason": str(e)}

class DialReader:
    def __init__(self, config_path="config.json"):
        print("Initializing Dial Reader...")
        self.config = self._load_config(config_path)
        self.device = self._get_device()
        self.yolo_model = self._load_yolo_model()
        self.classic_finder = _ClassicNeedleFinder()
        self.ml_finder = MLNeedleFinder(self.config, self.device)
        print("Dial Reader initialized.")

    def _load_config(self, config_path):
        if not Path(config_path).exists(): raise FileNotFoundError(f"Config file '{config_path}' not found.")
        with open(config_path, 'r') as f: return json.load(f)

    def _get_device(self):
        req_device = self.config.get("device", "auto").lower()
        if req_device == "auto": return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(req_device)

    def _load_yolo_model(self):
        model_config = self.config.get("models", {})
        model_path = model_config.get("model_path", "models/yolov5s.pt")
        if not Path(model_path).exists(): raise FileNotFoundError(f"Model file '{model_path}' not found.")
        
        print(f"Loading YOLO model from '{model_path}' onto {self.device}...")
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, _verbose=False)
        model.to(self.device)
        model.conf = model_config.get("confidence_threshold", 0.4)
        model.iou = model_config.get("iou_threshold", 0.5)
        model.classes = [0, 1]
        class_names = model_config.get('class_names', {'0':'dial', '1':'needle'})
        print(f"YOLO model loaded. Confidence: {model.conf}, Classes: {class_names}")
        return model

    def update_calibration(self, min_angle, max_angle, min_value, max_value):
        self.classic_finder.update_calibration(min_angle, max_angle, min_value, max_value)
        self.ml_finder.update_calibration(min_angle, max_angle, min_value, max_value)

    def detect(self, image):
        results = self.yolo_model(image)
        preds = results.pandas().xyxy[0]

        dials = preds[preds['name'] == 'dial']
        needles = preds[preds['name'] == 'needle']

        readings = []
        for _, dial_pred in dials.iterrows():
            x1, y1, x2, y2 = int(dial_pred['xmin']), int(dial_pred['ymin']), int(dial_pred['xmax']), int(dial_pred['ymax'])
            dial_crop = image[y1:y2, x1:x2]
            if dial_crop.size == 0: continue

            needle_result = self.ml_finder.find_needle(dial_crop)
            finder_method = "ml"

            if not needle_result["detected"]:
                needle_result = self.classic_finder.find_needle(dial_crop)
                finder_method = "classical"

            pointer_bbox = None
            for _, needle_pred in needles.iterrows():
                nx1, ny1, nx2, ny2 = int(needle_pred['xmin']), int(needle_pred['ymin']), int(needle_pred['xmax']), int(needle_pred['ymax'])
                if (x1 < (nx1+nx2)/2 < x2) and (y1 < (ny1+ny2)/2 < y2):
                    pointer_bbox = {"x": nx1, "y": ny1, "w": nx2 - nx1, "h": ny2 - ny1}
                    break

            if needle_result["detected"]:
                readings.append({
                    "value": needle_result["reading"],
                    "confidence": float(dial_pred['confidence']) * needle_result["confidence"],
                    "angle": needle_result.get("angle"),
                    "bounds": {"x": x1, "y": y1, "w": x2 - x1, "h": y2 - y1},
                    "pointer_bounds": pointer_bbox,
                    "finder_method": finder_method
                })
        
        return sorted(readings, key=lambda r: r['confidence'], reverse=True)
