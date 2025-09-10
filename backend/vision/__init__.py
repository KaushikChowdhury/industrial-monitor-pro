"""Computer-vision helpers for dial detection and reading.

Modules:
- dial_detector: find dial ROI using classical CV (Hough/contours)
- needle_reader: determine needle angle and scaled engineering value
- stabilize: light normalization and simple ROI stabilization hooks
- anomaly: simple online anomaly detectors (EWMA/CUSUM/ROC)
"""

