import cv2
import numpy as np
from typing import Optional, Tuple


def find_dial_roi(frame: np.ndarray) -> Tuple[Optional[Tuple[int,int,int,int]], Optional[Tuple[int,int,int]]]:
    """Find a circular dial ROI.

    Returns ((x0,y0,x1,y1), (cx,cy,r)) or (None, None) if not found.
    """
    if frame is None or frame.size == 0:
        return None, None

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Equalize/blur for robust circle edge detection
    gray = cv2.equalizeHist(gray)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=80,
        param1=120, param2=40, minRadius=60, maxRadius=0
    )
    if circles is not None and len(circles[0]) > 0:
        x, y, r = np.uint16(np.around(circles[0][0]))
        r_pad = int(r * 1.1)
        x0, y0 = max(0, x - r_pad), max(0, y - r_pad)
        x1, y1 = min(frame.shape[1], x + r_pad), min(frame.shape[0], y + r_pad)
        return (x0, y0, x1, y1), (int(x), int(y), int(r))

    # Fallback: largest circular-ish contour -> fit ellipse
    edges = cv2.Canny(gray, 50, 150)
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, None
    cnt = max(cnts, key=cv2.contourArea)
    if len(cnt) < 5:
        return None, None
    (cx, cy), (ma, mi), _ = cv2.fitEllipse(cnt)
    r = int(max(ma, mi) / 2)
    x0, y0 = max(0, int(cx - r)), max(0, int(cy - r))
    x1, y1 = min(frame.shape[1], int(cx + r)), min(frame.shape[0], int(cy + r))
    return (x0, y0, x1, y1), (int(cx), int(cy), int(r))

