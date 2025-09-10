import cv2
import numpy as np
from typing import Tuple


def normalize_lighting(roi_bgr: np.ndarray) -> np.ndarray:
    """Normalize lighting in ROI using CLAHE on L channel (LAB)."""
    if roi_bgr is None or roi_bgr.size == 0:
        return roi_bgr
    lab = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


class SimpleROITracker:
    """Tiny wrapper for OpenCV CSRT tracker to keep ROI stable between frames."""

    def __init__(self) -> None:
        self._tracker = None
        self._bbox = None

    def init(self, frame: np.ndarray, bbox_xywh: Tuple[int, int, int, int]) -> None:
        try:
            self._tracker = cv2.legacy.TrackerCSRT_create()
        except Exception:
            # Fallback for newer API path
            self._tracker = cv2.TrackerCSRT_create()
        self._bbox = bbox_xywh
        self._tracker.init(frame, tuple(map(float, bbox_xywh)))

    def update(self, frame: np.ndarray) -> Tuple[bool, Tuple[int, int, int, int]]:
        if self._tracker is None:
            return False, (0, 0, 0, 0)
        ok, box = self._tracker.update(frame)
        if not ok:
            return False, (0, 0, 0, 0)
        x, y, w, h = map(int, box)
        self._bbox = (x, y, w, h)
        return True, self._bbox

