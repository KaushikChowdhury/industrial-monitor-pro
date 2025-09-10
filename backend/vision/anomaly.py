from collections import deque
import math
import time
from typing import Deque, List, Tuple


class AnomalyState:
    def __init__(self, ewma_alpha: float = 0.2, win: int = 60):
        self.mu = None
        self.var = None
        self.alpha = ewma_alpha
        self.last = None
        self.cusum_pos = 0.0
        self.cusum_neg = 0.0
        self.hist: Deque[Tuple[float, float]] = deque(maxlen=win)

    def update(self, v: float) -> None:
        if self.mu is None:
            self.mu, self.var = v, 1e-6
        else:
            self.mu = self.alpha * v + (1 - self.alpha) * self.mu
            self.var = self.alpha * (v - self.mu) ** 2 + (1 - self.alpha) * self.var
        self.hist.append((time.time(), v))


def check_anomaly(state: AnomalyState, v: float, cfg: dict) -> List[Tuple[str, float]]:
    """Return list of (kind, value) anomalies.

    cfg keys: low_warn, high_warn, low_crit, high_crit, roc_limit, cusum_k, cusum_h
    """
    state.update(v)
    mu = float(state.mu) if state.mu is not None else v
    sigma = math.sqrt(max(float(state.var or 0.0), 1e-6))
    issues: List[Tuple[str, float]] = []

    if v < cfg.get("low_crit", -math.inf) or v > cfg.get("high_crit", math.inf):
        issues.append(("CRIT_THRESHOLD", v))
    elif v < cfg.get("low_warn", -math.inf) or v > cfg.get("high_warn", math.inf):
        issues.append(("WARN_THRESHOLD", v))

    # Rate-of-change over last sample
    if state.last is not None and len(state.hist) >= 2:
        t0, _ = state.hist[-2]
        t1, _ = state.hist[-1]
        dt = max(1e-3, t1 - t0)
        roc = abs(v - state.last) / dt
        if roc > float(cfg.get("roc_limit", math.inf)):
            issues.append(("RATE_OF_CHANGE", roc))

    # Two-sided CUSUM
    k = float(cfg.get("cusum_k", 0.5 * sigma))
    h = float(cfg.get("cusum_h", 5.0 * sigma))
    state.cusum_pos = max(0.0, state.cusum_pos + (v - (mu + k)))
    state.cusum_neg = max(0.0, state.cusum_neg + ((mu - k) - v))
    if state.cusum_pos > h:
        issues.append(("CUSUM_POS", state.cusum_pos))
    if state.cusum_neg > h:
        issues.append(("CUSUM_NEG", state.cusum_neg))

    state.last = v
    return issues

