import sqlite3
from typing import Any, Dict, Iterable, List, Optional, Tuple


SCHEMA = """
CREATE TABLE IF NOT EXISTS gauges (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  name TEXT,
  camera_id TEXT,
  roi_x0 INTEGER, roi_y0 INTEGER, roi_x1 INTEGER, roi_y1 INTEGER,
  min_angle REAL, max_angle REAL, min_value REAL, max_value REAL,
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS gauge_thresholds (
  gauge_id INTEGER PRIMARY KEY,
  low_warn REAL, high_warn REAL,
  low_crit REAL, high_crit REAL,
  roc_limit REAL,
  cusum_k REAL, cusum_h REAL,
  FOREIGN KEY(gauge_id) REFERENCES gauges(id)
);

CREATE TABLE IF NOT EXISTS gauge_readings (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  gauge_id INTEGER NOT NULL,
  ts DATETIME DEFAULT CURRENT_TIMESTAMP,
  angle REAL, value REAL,
  FOREIGN KEY(gauge_id) REFERENCES gauges(id)
);

CREATE TABLE IF NOT EXISTS gauge_events (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  gauge_id INTEGER NOT NULL,
  ts DATETIME DEFAULT CURRENT_TIMESTAMP,
  severity TEXT,  -- INFO/WARN/CRIT
  kind TEXT,      -- e.g., RATE_OF_CHANGE, CUSUM_POS
  message TEXT,
  value REAL,
  FOREIGN KEY(gauge_id) REFERENCES gauges(id)
);
"""


def init_models(db_path: str) -> None:
    conn = sqlite3.connect(db_path)
    conn.executescript(SCHEMA)
    conn.commit()
    conn.close()


def upsert_gauge(db_path: str, name: str, camera_id: str,
                 roi: Optional[Tuple[int, int, int, int]] = None,
                 calibration: Optional[Dict[str, float]] = None,
                 gauge_id: Optional[int] = None) -> int:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    if gauge_id is None:
        cur.execute(
            "INSERT INTO gauges(name, camera_id, roi_x0, roi_y0, roi_x1, roi_y1, min_angle, max_angle, min_value, max_value) VALUES(?,?,?,?,?,?,?,?,?,?)",
            (
                name, camera_id,
                *(roi or (None, None, None, None)),
                *(None if calibration is None else (
                    calibration.get("min_angle"), calibration.get("max_angle"),
                    calibration.get("min_value"), calibration.get("max_value")
                ))
            )
        )
        gauge_id = int(cur.lastrowid)
    else:
        x0, y0, x1, y1 = roi or (None, None, None, None)
        min_a = calibration.get("min_angle") if calibration else None
        max_a = calibration.get("max_angle") if calibration else None
        min_v = calibration.get("min_value") if calibration else None
        max_v = calibration.get("max_value") if calibration else None
        cur.execute(
            """
            UPDATE gauges SET
              name = COALESCE(?, name), camera_id = COALESCE(?, camera_id),
              roi_x0 = COALESCE(?, roi_x0), roi_y0 = COALESCE(?, roi_y0),
              roi_x1 = COALESCE(?, roi_x1), roi_y1 = COALESCE(?, roi_y1),
              min_angle = COALESCE(?, min_angle), max_angle = COALESCE(?, max_angle),
              min_value = COALESCE(?, min_value), max_value = COALESCE(?, max_value),
              updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (name, camera_id, x0, y0, x1, y1, min_a, max_a, min_v, max_v, gauge_id)
        )
    conn.commit(); conn.close()
    return int(gauge_id)


def set_gauge_calibration(db_path: str, gauge_id: int, min_angle: float, max_angle: float, min_value: float, max_value: float) -> None:
    conn = sqlite3.connect(db_path)
    conn.execute(
        "UPDATE gauges SET min_angle=?, max_angle=?, min_value=?, max_value=?, updated_at=CURRENT_TIMESTAMP WHERE id=?",
        (min_angle, max_angle, min_value, max_value, gauge_id)
    )
    conn.commit(); conn.close()


def set_gauge_thresholds(db_path: str, gauge_id: int, low_warn: float, high_warn: float, low_crit: float, high_crit: float,
                         roc_limit: float, cusum_k: float, cusum_h: float) -> None:
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        INSERT INTO gauge_thresholds(gauge_id, low_warn, high_warn, low_crit, high_crit, roc_limit, cusum_k, cusum_h)
        VALUES(?,?,?,?,?,?,?,?)
        ON CONFLICT(gauge_id) DO UPDATE SET
          low_warn=excluded.low_warn, high_warn=excluded.high_warn,
          low_crit=excluded.low_crit, high_crit=excluded.high_crit,
          roc_limit=excluded.roc_limit, cusum_k=excluded.cusum_k, cusum_h=excluded.cusum_h
        """,
        (gauge_id, low_warn, high_warn, low_crit, high_crit, roc_limit, cusum_k, cusum_h)
    )
    conn.commit(); conn.close()


def get_gauge_thresholds(db_path: str, gauge_id: int) -> Optional[dict]:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT low_warn, high_warn, low_crit, high_crit, roc_limit, cusum_k, cusum_h FROM gauge_thresholds WHERE gauge_id=?", (gauge_id,))
    row = cur.fetchone(); conn.close()
    if not row:
        return None
    return {
        "low_warn": float(row[0]), "high_warn": float(row[1]),
        "low_crit": float(row[2]), "high_crit": float(row[3]),
        "roc_limit": float(row[4]), "cusum_k": float(row[5]), "cusum_h": float(row[6]),
    }


def insert_gauge_reading(db_path: str, gauge_id: int, angle: float, value: float) -> None:
    conn = sqlite3.connect(db_path)
    conn.execute("INSERT INTO gauge_readings(gauge_id, angle, value) VALUES(?,?,?)", (gauge_id, angle, value))
    conn.commit(); conn.close()


def recent_gauge_readings(db_path: str, gauge_id: int, since: Optional[str] = None) -> List[dict]:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    if since:
        cur.execute("SELECT ts, angle, value FROM gauge_readings WHERE gauge_id=? AND ts > ? ORDER BY ts ASC", (gauge_id, since))
    else:
        cur.execute("SELECT ts, angle, value FROM gauge_readings WHERE gauge_id=? ORDER BY ts DESC LIMIT 500", (gauge_id,))
    rows = [{"ts": r[0], "angle": r[1], "value": r[2]} for r in cur.fetchall()]
    conn.close(); return rows


def insert_gauge_event(db_path: str, gauge_id: int, severity: str, kind: str, message: str, value: float) -> None:
    conn = sqlite3.connect(db_path)
    conn.execute("INSERT INTO gauge_events(gauge_id, severity, kind, message, value) VALUES(?,?,?,?,?)",
                 (gauge_id, severity, kind, message, value))
    conn.commit(); conn.close()


def recent_events(db_path: str, severity: Optional[str] = None, limit: int = 200) -> List[dict]:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    if severity:
        cur.execute("SELECT id, gauge_id, ts, severity, kind, message, value FROM gauge_events WHERE severity=? ORDER BY ts DESC LIMIT ?",
                    (severity, limit))
    else:
        cur.execute("SELECT id, gauge_id, ts, severity, kind, message, value FROM gauge_events ORDER BY ts DESC LIMIT ?",
                    (limit,))
    rows = [
        {"id": r[0], "gauge_id": r[1], "ts": r[2], "severity": r[3], "kind": r[4], "message": r[5], "value": r[6]}
        for r in cur.fetchall()
    ]
    conn.close(); return rows

