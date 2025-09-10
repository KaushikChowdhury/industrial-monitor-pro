import cv2
import numpy as np
from dataclasses import dataclass

@dataclass
class DialDetector:
    min_angle: float = 0.0
    max_angle: float = 270.0
    min_value: float = 0.0
    max_value: float = 100.0

    def detect_dial(self, image):
        try:
            # Preprocess: denoise + local contrast to handle glare/low contrast
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            gray = cv2.bilateralFilter(gray, d=7, sigmaColor=75, sigmaSpace=75)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray)

            circles = cv2.HoughCircles(
                gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=80,
                param1=120, param2=40, minRadius=30, maxRadius=0
            )
            if circles is None:
                return {"detected": False, "reading": 0.0, "confidence": 0.0}

            circles = np.uint16(np.around(circles))
            cx, cy, r = sorted(circles[0, :], key=lambda c: c[2], reverse=True)[0]

            mask = np.zeros_like(gray)
            cv2.circle(mask, (cx, cy), r, 255, -1)
            dial_roi = cv2.bitwise_and(gray, mask)
            # Use an annulus to avoid labels near center and outside noise
            inner_r = max(5, int(r * 0.2))
            cv2.circle(mask, (cx, cy), inner_r, 0, -1)
            dial_roi = cv2.bitwise_and(gray, mask)

            # Auto-Canny thresholds based on median
            v = np.median(dial_roi[dial_roi > 0]) if np.any(dial_roi > 0) else 0
            lower = int(max(0, 0.66 * v))
            upper = int(min(255, 1.33 * v))
            edges = cv2.Canny(dial_roi, lower if lower>0 else 30, upper if upper>0 else 120)

            # Try colored-needle detection (e.g., red/black)
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            # Red can wrap around hue: combine two ranges
            red1 = cv2.inRange(hsv, (0, 70, 50), (10, 255, 255))
            red2 = cv2.inRange(hsv, (170, 70, 50), (180, 255, 255))
            red = cv2.bitwise_or(red1, red2)
            black = cv2.inRange(hsv, (0, 0, 0), (180, 255, 50))
            color_mask = cv2.bitwise_or(red, black)
            color_mask = cv2.bitwise_and(color_mask, mask)

            edges_color = cv2.Canny(color_mask, 50, 150)
            edges_all = cv2.bitwise_or(edges, edges_color)

            lines = cv2.HoughLinesP(edges_all, 1, np.pi/180, threshold=60, minLineLength=int(r*0.4), maxLineGap=12)

            angle = None; best_len = 0
            if lines is not None:
                for l in lines:
                    x1, y1, x2, y2 = l[0]
                    # Prefer lines that originate near center and extend outward
                    d1 = ((x1-cx)**2 + (y1-cy)**2) ** 0.5
                    d2 = ((x2-cx)**2 + (y2-cy)**2) ** 0.5
                    near_center = min(d1, d2) < r * 0.35
                    reaches_out = max(d1, d2) > r * 0.6
                    if not (near_center and reaches_out):
                        continue
                    # Use center->far endpoint direction for angle stability
                    if d1 > d2:
                        fx, fy = x1, y1
                    else:
                        fx, fy = x2, y2
                    length = abs(max(d1, d2) - min(d1, d2))
                    if length > best_len:
                        best_len = length
                        angle = (np.degrees(np.arctan2(fy - cy, fx - cx)) + 360) % 360

            if angle is None:
                angle = 0.5*(self.min_angle + self.max_angle)

            sweep = (angle - self.min_angle) % 360
            total_span = (self.max_angle - self.min_angle) % 360
            if total_span == 0:
                total_span = 270
            norm = np.clip(sweep/total_span, 0, 1)
            value = self.min_value + norm * (self.max_value - self.min_value)

            return {
                "detected": True,
                "center": (int(cx), int(cy)),
                "radius": int(r),
                "angle": float(angle),
                "reading": float(value),
                "confidence": 0.9 if lines is not None else 0.5
            }
        except Exception:
            return {"detected": False, "reading": 0.0, "confidence": 0.0}
