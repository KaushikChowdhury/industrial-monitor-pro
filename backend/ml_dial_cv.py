"""
Enhanced ML-Powered Dial Detection System
Integrates YOLO, VDN, and classical CV with few-shot learning capabilities
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
import json
import os
from pathlib import Path
import requests
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============= Model Download Manager =============
class ModelManager:
    """Manages downloading and caching of pre-trained models"""

    MODEL_URLS = {
        'yolov5s_gauge': 'https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt',
        'vdn_resnet34': 'https://github.com/DrawZeroPoint/VectorDetectionNetwork/releases/download/v1.0/vdn_best.pth.tar',
        'gauge_detector': 'https://huggingface.co/spaces/gauge-reading/resolve/main/gauge_detector.pt'
    }

    def __init__(self, cache_dir: str = "./weights"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

    def download_model(self, model_name: str) -> Path:
        """Download model if not cached"""
        model_path = self.cache_dir / f"{model_name}.pt"

        if not model_path.exists():
            url = self.MODEL_URLS.get(model_name)
            if not url:
                logger.warning(f"No URL for {model_name}, using placeholder")
                # Create placeholder for testing
                torch.save({'state_dict': {}}, model_path)
                return model_path

            logger.info(f"Downloading {model_name} from {url}")
            response = requests.get(url, stream=True)
            total_size = int(response.headers.get('content-length', 0))

            with open(model_path, 'wb') as f:
                with tqdm(total=total_size, unit='B', unit_scale=True) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                        pbar.update(len(chunk))

        return model_path


# ============= VDN Implementation =============
class VectorDetectionNetwork(nn.Module):
    """Simplified VDN for pointer detection as vectors"""

    def __init__(self, backbone='resnet34', input_size=384):
        super().__init__()
        self.input_size = input_size

        # Backbone (ResNet variants)
        if backbone == 'resnet34':
            from torchvision.models import resnet34
            self.backbone = resnet34(pretrained=True)
            self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])
            channels = 512
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # Detection heads
        self.confidence_head = nn.Sequential(
            nn.Conv2d(channels, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 1, 1),
            nn.Sigmoid()
        )

        self.vector_head = nn.Sequential(
            nn.Conv2d(channels, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 2, 1)  # x, y components
        )

    def forward(self, x):
        features = self.backbone(x)
        confidence = self.confidence_head(features)
        vectors = self.vector_head(features)
        return confidence, vectors

    def detect_pointer(self, image: np.ndarray) -> Dict:
        """Detect pointer as vector"""
        # Preprocess
        img_tensor = self._preprocess(image)

        with torch.no_grad():
            confidence, vectors = self(img_tensor)

        # Find peak in confidence map
        conf_np = confidence[0, 0].cpu().numpy()
        peak_idx = np.unravel_index(conf_np.argmax(), conf_np.shape)

        # Get vector at peak
        vec_x = vectors[0, 0, peak_idx[0], peak_idx[1]].item()
        vec_y = vectors[0, 1, peak_idx[0], peak_idx[1]].item()

        # Convert to angle
        angle = np.degrees(np.arctan2(vec_y, vec_x)) % 360

        return {
            'angle': angle,
            'confidence': conf_np[peak_idx],
            'center': peak_idx
        }

    def _preprocess(self, image):
        img = cv2.resize(image, (self.input_size, self.input_size))
        img = img.transpose(2, 0, 1) / 255.0
        return torch.FloatTensor(img).unsqueeze(0)


# ============= YOLO-based Gauge Detector =============
class YOLOGaugeDetector:
    """YOLO-based detection for gauge and pointer"""

    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if model_path and Path(model_path).exists():
            try:
                import yolov5
                self.model = yolov5.load(model_path)
                self.model.to(self.device)
            except ImportError:
                logger.warning("YOLOv5 not installed, using fallback")

    def detect_gauge_components(self, image: np.ndarray) -> Dict:
        """Detect gauge, pointer, and scale marks"""
        if self.model is None:
            # Fallback to classical detection
            return self._classical_detection(image)

        results = self.model(image)
        detections = results.pandas().xyxy[0]

        gauge_info = {
            'gauge_bbox': None,
            'pointer_bbox': None,
            'scale_marks': [],
            'confidence': 0.0
        }

        for _, det in detections.iterrows():
            if det['name'] == 'gauge':
                gauge_info['gauge_bbox'] = [det['xmin'], det['ymin'], det['xmax'], det['ymax']]
                gauge_info['confidence'] = det['confidence']
            elif det['name'] == 'pointer':
                gauge_info['pointer_bbox'] = [det['xmin'], det['ymin'], det['xmax'], det['ymax']]
            elif det['name'] == 'scale':
                gauge_info['scale_marks'].append([det['xmin'], det['ymin'], det['xmax'], det['ymax']])

        return gauge_info

    def _classical_detection(self, image):
        """Fallback classical detection"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=80,
                                   param1=80, param2=30, minRadius=30, maxRadius=300)

        if circles is not None:
            circle = circles[0, 0]
            return {
                'gauge_bbox': [circle[0] - circle[2], circle[1] - circle[2],
                               circle[0] + circle[2], circle[1] + circle[2]],
                'pointer_bbox': None,
                'scale_marks': [],
                'confidence': 0.7
            }
        return {'gauge_bbox': None, 'pointer_bbox': None, 'scale_marks': [], 'confidence': 0.0}


# ============= Enhanced ML Dial Detector =============
@dataclass
class MLDialDetector:
    """Main ML-powered dial detector with multiple backend support"""

    min_angle: float = 0.0
    max_angle: float = 270.0
    min_value: float = 0.0
    max_value: float = 100.0
    backend: str = 'hybrid'  # 'yolo', 'vdn', 'classical', 'hybrid'
    enable_tracking: bool = True

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

        self.model_manager = ModelManager()
        self.yolo_detector = None
        self.vdn_detector = None
        self.calibration_points = []
        self.tracking_history = []

        # Initialize models based on backend
        self._initialize_models()

    def _initialize_models(self):
        """Initialize ML models"""
        if self.backend in ['yolo', 'hybrid']:
            try:
                model_path = self.model_manager.download_model('yolov5s_gauge')
                self.yolo_detector = YOLOGaugeDetector(str(model_path))
            except Exception as e:
                logger.warning(f"Failed to load YOLO: {e}")

        if self.backend in ['vdn', 'hybrid']:
            try:
                self.vdn_detector = VectorDetectionNetwork()
                # Load pre-trained weights if available
                model_path = self.model_manager.download_model('vdn_resnet34')
                if model_path.exists():
                    checkpoint = torch.load(model_path, map_location='cpu')
                    if 'state_dict' in checkpoint:
                        self.vdn_detector.load_state_dict(checkpoint['state_dict'], strict=False)
            except Exception as e:
                logger.warning(f"Failed to load VDN: {e}")

    def detect_dial(self, image: np.ndarray) -> Dict:
        """Main detection method with multi-backend support"""
        results = []

        # YOLO detection
        if self.yolo_detector and self.backend in ['yolo', 'hybrid']:
            yolo_result = self._detect_with_yolo(image)
            if yolo_result['detected']:
                results.append(yolo_result)

        # VDN detection
        if self.vdn_detector and self.backend in ['vdn', 'hybrid']:
            vdn_result = self._detect_with_vdn(image)
            if vdn_result['detected']:
                results.append(vdn_result)

        # Classical detection as fallback
        if not results or self.backend == 'classical':
            classical_result = self._detect_classical(image)
            results.append(classical_result)

        # Ensemble results if multiple detections
        final_result = self._ensemble_results(results)

        # Apply temporal smoothing if tracking enabled
        if self.enable_tracking:
            final_result = self._apply_temporal_smoothing(final_result)

        return final_result

    def _detect_with_yolo(self, image: np.ndarray) -> Dict:
        """YOLO-based detection"""
        components = self.yolo_detector.detect_gauge_components(image)

        if components['gauge_bbox'] is None:
            return {'detected': False, 'reading': 0.0, 'confidence': 0.0}

        # Extract gauge region
        bbox = components['gauge_bbox']
        gauge_roi = image[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]

        # Detect pointer angle
        angle = self._detect_pointer_angle(gauge_roi, components.get('pointer_bbox'))

        # Convert angle to reading
        reading = self._angle_to_reading(angle)

        return {
            'detected': True,
            'center': ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2),
            'radius': max(bbox[2] - bbox[0], bbox[3] - bbox[1]) / 2,
            'angle': angle,
            'reading': reading,
            'confidence': components['confidence'],
            'method': 'yolo'
        }

    def _detect_with_vdn(self, image: np.ndarray) -> Dict:
        """VDN-based detection"""
        result = self.vdn_detector.detect_pointer(image)

        if result['confidence'] < 0.5:
            return {'detected': False, 'reading': 0.0, 'confidence': 0.0}

        reading = self._angle_to_reading(result['angle'])

        return {
            'detected': True,
            'center': result['center'],
            'radius': 100,  # Estimated
            'angle': result['angle'],
            'reading': reading,
            'confidence': result['confidence'],
            'method': 'vdn'
        }

    def _detect_classical(self, image: np.ndarray) -> Dict:
        """Classical CV detection (improved)"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Multi-scale circle detection
        scales = [1.0, 1.2, 1.5]
        best_circle = None
        best_score = 0

        for scale in scales:
            scaled_gray = cv2.resize(gray, None, fx=scale, fy=scale)
            circles = cv2.HoughCircles(
                scaled_gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=80,
                param1=80, param2=30, minRadius=30, maxRadius=300
            )

            if circles is not None:
                for circle in circles[0]:
                    score = self._evaluate_circle_quality(scaled_gray, circle / scale)
                    if score > best_score:
                        best_score = score
                        best_circle = circle / scale

        if best_circle is None:
            return {'detected': False, 'reading': 0.0, 'confidence': 0.0}

        cx, cy, r = best_circle

        # Enhanced pointer detection with multiple methods
        angle = self._detect_pointer_multimethod(gray, (cx, cy), r)
        reading = self._angle_to_reading(angle)

        return {
            'detected': True,
            'center': (int(cx), int(cy)),
            'radius': int(r),
            'angle': angle,
            'reading': reading,
            'confidence': min(best_score, 0.9),
            'method': 'classical'
        }

    def _detect_pointer_multimethod(self, gray: np.ndarray, center: Tuple, radius: float) -> float:
        """Detect pointer using multiple methods and vote"""
        angles = []
        weights = []

        # Method 1: Line detection
        angle1 = self._detect_pointer_lines(gray, center, radius)
        if angle1 is not None:
            angles.append(angle1)
            weights.append(0.4)

        # Method 2: Radial intensity analysis
        angle2 = self._detect_pointer_radial(gray, center, radius)
        if angle2 is not None:
            angles.append(angle2)
            weights.append(0.3)

        # Method 3: Template matching
        angle3 = self._detect_pointer_template(gray, center, radius)
        if angle3 is not None:
            angles.append(angle3)
            weights.append(0.3)

        if not angles:
            return 135.0  # Default middle position

        # Weighted average
        weighted_angle = np.average(angles, weights=weights[:len(angles)])
        return weighted_angle

    def _detect_pointer_lines(self, gray, center, radius):
        """Line-based pointer detection"""
        cx, cy = int(center[0]), int(center[1])
        r = int(radius)

        # Create mask for dial region
        mask = np.zeros_like(gray)
        cv2.circle(mask, (cx, cy), r, 255, -1)

        # Edge detection
        edges = cv2.Canny(gray & mask, 50, 150)

        # Line detection
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=30,
                                minLineLength=int(r * 0.3), maxLineGap=10)

        if lines is None:
            return None

        # Find line passing through center
        best_angle = None
        min_dist = float('inf')

        for line in lines:
            x1, y1, x2, y2 = line[0]

            # Distance from center to line
            dist = abs((y2 - y1) * cx - (x2 - x1) * cy + x2 * y1 - y2 * x1) / np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)

            if dist < min_dist and dist < r * 0.2:
                min_dist = dist
                best_angle = np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 360

        return best_angle

    def _detect_pointer_radial(self, gray, center, radius):
        """Radial intensity-based detection"""
        cx, cy = int(center[0]), int(center[1])
        r = int(radius)

        angles = np.linspace(0, 360, 360)
        intensities = []

        for angle in angles:
            rad = np.radians(angle)

            # Sample along radial line
            points = []
            for dist in np.linspace(r * 0.2, r * 0.8, 20):
                x = int(cx + dist * np.cos(rad))
                y = int(cy + dist * np.sin(rad))

                if 0 <= x < gray.shape[1] and 0 <= y < gray.shape[0]:
                    points.append(gray[y, x])

            if points:
                # Variance of intensities (pointer has consistent dark values)
                intensities.append(np.var(points))
            else:
                intensities.append(float('inf'))

        # Find angle with minimum variance
        if intensities:
            min_idx = np.argmin(intensities)
            return angles[min_idx]

        return None

    def _detect_pointer_template(self, gray, center, radius):
        """Template matching for pointer"""
        # Create synthetic pointer template
        template_size = int(radius * 0.6)
        template = np.zeros((template_size, 10), dtype=np.uint8)
        cv2.rectangle(template, (2, 0), (8, template_size), 0, -1)

        best_angle = None
        best_score = 0

        cx, cy = int(center[0]), int(center[1])
        r = int(radius)

        # ROI around dial
        x1, y1 = max(0, cx - r), max(0, cy - r)
        x2, y2 = min(gray.shape[1], cx + r), min(gray.shape[0], cy + r)
        roi = gray[y1:y2, x1:x2]

        for angle in range(0, 360, 5):
            # Rotate template
            M = cv2.getRotationMatrix2D((template.shape[1] / 2, template.shape[0] / 2), angle, 1)
            rotated = cv2.warpAffine(template, M, (template.shape[1], template.shape[0]))

            # Match
            result = cv2.matchTemplate(roi, rotated, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(result)

            if max_val > best_score:
                best_score = max_val
                best_angle = angle

        return best_angle if best_score > 0.3 else None

    def _evaluate_circle_quality(self, gray, circle):
        """Evaluate how good a detected circle is"""
        cx, cy, r = int(circle[0]), int(circle[1]), int(circle[2])

        # Check if circle is within image bounds
        if cx - r < 0 or cy - r < 0 or cx + r >= gray.shape[1] or cy + r >= gray.shape[0]:
            return 0.0

        # Create circular mask
        mask = np.zeros_like(gray)
        cv2.circle(mask, (cx, cy), r, 255, 2)

        # Check edge strength on circle perimeter
        edges = cv2.Canny(gray, 50, 150)
        edge_overlap = cv2.bitwise_and(edges, mask)
        edge_score = np.sum(edge_overlap) / (2 * np.pi * r * 255)

        return edge_score

    def _detect_pointer_angle(self, roi, pointer_bbox=None):
        """Detect pointer angle from ROI"""
        if pointer_bbox:
            # Use detected pointer bbox
            px1, py1, px2, py2 = pointer_bbox
            pointer_center = ((px1 + px2) / 2, (py1 + py2) / 2)
            angle = np.degrees(np.arctan2(py2 - py1, px2 - px1)) % 360
            return angle

        # Fallback to edge-based detection
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi
        return self._detect_pointer_multimethod(gray,
                                                (roi.shape[1] / 2, roi.shape[0] / 2),
                                                min(roi.shape) / 2)

    def _angle_to_reading(self, angle: float) -> float:
        """Convert angle to gauge reading"""
        # Normalize angle to gauge range
        sweep = (angle - self.min_angle) % 360
        total_span = (self.max_angle - self.min_angle) % 360

        if total_span == 0:
            total_span = 270

        norm = np.clip(sweep / total_span, 0, 1)
        return self.min_value + norm * (self.max_value - self.min_value)

    def _ensemble_results(self, results: List[Dict]) -> Dict:
        """Ensemble multiple detection results"""
        if not results:
            return {'detected': False, 'reading': 0.0, 'confidence': 0.0}

        if len(results) == 1:
            return results[0]

        # Weighted average based on confidence
        total_weight = sum(r['confidence'] for r in results if r['detected'])

        if total_weight == 0:
            return results[0]

        ensemble = {
            'detected': True,
            'reading': sum(r['reading'] * r['confidence'] for r in results if r['detected']) / total_weight,
            'confidence': max(r['confidence'] for r in results),
            'angle': sum(r.get('angle', 0) * r['confidence'] for r in results if r['detected']) / total_weight,
            'method': 'ensemble'
        }

        # Use center from highest confidence result
        best_result = max(results, key=lambda r: r.get('confidence', 0))
        ensemble['center'] = best_result.get('center', (0, 0))
        ensemble['radius'] = best_result.get('radius', 100)

        return ensemble

    def _apply_temporal_smoothing(self, result: Dict) -> Dict:
        """Apply temporal smoothing to reduce noise"""
        if not result['detected']:
            return result

        self.tracking_history.append(result['reading'])

        # Keep last N readings
        window_size = 5
        if len(self.tracking_history) > window_size:
            self.tracking_history = self.tracking_history[-window_size:]

        if len(self.tracking_history) >= 3:
            # Median filter for outlier rejection
            smoothed_reading = np.median(self.tracking_history)
            result['reading'] = smoothed_reading
            result['smoothed'] = True

        return result

    def calibrate_interactive(self, image: np.ndarray, clicks: List[Tuple[int, int]]) -> bool:
        """Two-click calibration: click min value, then max value"""
        if len(clicks) < 2:
            return False

        # Get center from detection
        result = self.detect_dial(image)
        if not result['detected']:
            return False

        cx, cy = result['center']

        # Calculate angles for clicked points
        min_click = clicks[0]
        max_click = clicks[1]

        self.min_angle = np.degrees(np.arctan2(min_click[1] - cy, min_click[0] - cx)) % 360
        self.max_angle = np.degrees(np.arctan2(max_click[1] - cy, max_click[0] - cx)) % 360

        self.calibration_points = clicks
        logger.info(f"Calibrated: min_angle={self.min_angle:.1f}, max_angle={self.max_angle:.1f}")

        return True

    def set_value_range(self, min_val: float, max_val: float):
        """Set the value range for the gauge"""
        self.min_value = min_val
        self.max_value = max_val

    def few_shot_adapt(self, samples: List[Tuple[np.ndarray, float]]):
        """Few-shot adaptation with labeled samples"""
        if not samples or not self.vdn_detector:
            return

        # Simple few-shot learning: fine-tune last layer
        optimizer = torch.optim.Adam(self.vdn_detector.vector_head.parameters(), lr=0.001)

        for image, true_reading in samples:
            # Convert reading to angle
            norm = (true_reading - self.min_value) / (self.max_value - self.min_value)
            true_angle = self.min_angle + norm * (self.max_angle - self.min_angle)

            # Forward pass
            img_tensor = self.vdn_detector._preprocess(image)
            confidence, vectors = self.vdn_detector(img_tensor)

            # Compute loss
            predicted_angle = torch.atan2(vectors[0, 1], vectors[0, 0]) * 180 / np.pi
            loss = F.mse_loss(predicted_angle.mean(), torch.tensor(true_angle))

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        logger.info(f"Few-shot adaptation completed with {len(samples)} samples")


# ============= Export Enhanced Detector =============
def create_enhanced_detector(**kwargs) -> MLDialDetector:
    """Factory function to create enhanced detector"""
    return MLDialDetector(**kwargs)
