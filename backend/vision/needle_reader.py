import cv2
import numpy as np
from math import atan2, degrees
from typing import Dict, Optional, Tuple


def _dominant_line(binary: np.ndarray, center: Tuple[int, int]) -> Optional[Tuple[int, int, int, int]]:
    lines = cv2.HoughLinesP(binary, 1, np.pi / 180, threshold=60, minLineLength=40, maxLineGap=10)
    if lines is None:
        return None
    cx, cy = center
    best = None
    best_score = -1e9
    for x1, y1, x2, y2 in lines[:, 0, :]:
        vx, vy = x2 - x1, y2 - y1
        length = np.hypot(vx, vy)
        if length < 1:
            continue
        # distance from center to line
        dist = abs(np.cross([vx, vy], [cx - x1, cy - y1])) / (length + 1e-6)
        score = length - 0.5 * dist
        if score > best_score:
            best_score = score
            best = (x1, y1, x2, y2)
    return best


def read_angle_and_value(roi_bgr: np.ndarray, dial_center: Tuple[int, int], cal: Dict[str, float]) -> Tuple[Optional[float], Optional[float], Optional[Tuple[int, int]]]:
    """Return (angle_deg, value, tip_xy) or (None, None, None) if not found.

    cal keys: min_angle, max_angle, min_value, max_value (degrees CCW from +x)
    """
    if roi_bgr is None or roi_bgr.size == 0:
        return None, None, None
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(gray, 40, 120)
    dom = _dominant_line(edges, dial_center)
    if not dom:
        return None, None, None
    x1, y1, x2, y2 = dom

    cx, cy = dial_center
    d1 = (x1 - cx) ** 2 + (y1 - cy) ** 2
    d2 = (x2 - cx) ** 2 + (y2 - cy) ** 2
    tipx, tipy = (x1, y1) if d1 > d2 else (x2, y2)

    # 0° = +x, CCW positive
    angle = (degrees(atan2(cy - tipy, tipx - cx)) + 360) % 360

    a0, a1 = cal["min_angle"] % 360, cal["max_angle"] % 360
    v0, v1 = cal["min_value"], cal["max_value"]

    span = (a1 - a0 + 360) % 360
    span = max(span, 1e-6)
    da = (angle - a0 + 360) % 360
    value = v0 + (v1 - v0) * (da / span)
    return angle, value, (int(tipx), int(tipy))

