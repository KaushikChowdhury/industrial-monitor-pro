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
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (5, 5), 0)

            circles = cv2.HoughCircles(
                gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=80,
                param1=80, param2=30, minRadius=30, maxRadius=300
            )
            if circles is None:
                return {"detected": False, "reading": 0.0, "confidence": 0.0}

            circles = np.uint16(np.around(circles))
            cx, cy, r = sorted(circles[0, :], key=lambda c: c[2], reverse=True)[0]

            mask = np.zeros_like(gray)
            cv2.circle(mask, (cx, cy), r, 255, -1)
            dial_roi = cv2.bitwise_and(gray, mask)

            edges = cv2.Canny(dial_roi, 50, 150)
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=60, minLineLength=int(r*0.4), maxLineGap=12)

            angle = None; best_len = 0
            if lines is not None:
                for l in lines:
                    x1, y1, x2, y2 = l[0]
                    # Prefer lines that involve center region
                    if (x1-cx)**2 + (y1-cy)**2 > (r*0.7)**2 and (x2-cx)**2 + (y2-cy)**2 > (r*0.7)**2:
                        continue
                    length = ((x2-x1)**2 + (y2-y1)**2) ** 0.5
                    if length > best_len:
                        best_len = length
                        angle = (np.degrees(np.arctan2(y2-y1, x2-x1)) + 360) % 360

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
                "confidence": 0.85 if lines is not None else 0.5
            }
        except Exception:
            return {"detected": False, "reading": 0.0, "confidence": 0.0}