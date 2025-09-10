import sqlite3

def init_database(path: str):
    conn = sqlite3.connect(path)
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS dial_readings (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        camera_id TEXT NOT NULL,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        reading REAL NOT NULL,
        is_anomaly INTEGER DEFAULT 0,
        confidence REAL DEFAULT 0.0,
        detected INTEGER DEFAULT 1
    )""")

    # Backward-compatible migration: add detected column if missing
    try:
        cur.execute("ALTER TABLE dial_readings ADD COLUMN detected INTEGER DEFAULT 1")
    except Exception:
        pass

    cur.execute("""
    CREATE TABLE IF NOT EXISTS anomalies (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        camera_id TEXT NOT NULL,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        reading REAL NOT NULL,
        threshold REAL NOT NULL,
        severity TEXT DEFAULT 'LOW',
        resolved INTEGER DEFAULT 0
    )""")

    cur.execute("""
    CREATE TABLE IF NOT EXISTS camera_status (
        camera_id TEXT PRIMARY KEY,
        is_active INTEGER DEFAULT 1,
        last_seen DATETIME DEFAULT CURRENT_TIMESTAMP,
        total_anomalies INTEGER DEFAULT 0,
        avg_reading REAL DEFAULT 0.0
    )""")

    # Per-camera thresholds (override global)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS camera_thresholds (
        camera_id TEXT PRIMARY KEY,
        low REAL NOT NULL,
        med REAL NOT NULL,
        high REAL NOT NULL
    )""")

    conn.commit(); conn.close()


def insert_reading(path: str, camera_id: str, reading: float, is_anomaly: int, confidence: float, detected: int = 1):
    conn = sqlite3.connect(path); cur = conn.cursor()
    cur.execute("INSERT INTO dial_readings(camera_id, reading, is_anomaly, confidence, detected) VALUES(?,?,?,?,?)",
                (camera_id, reading, is_anomaly, confidence, detected))
    conn.commit(); conn.close()


def insert_anomaly(path: str, camera_id: str, reading: float, threshold: float, severity: str):
    conn = sqlite3.connect(path); cur = conn.cursor()
    cur.execute("INSERT INTO anomalies(camera_id, reading, threshold, severity) VALUES(?,?,?,?)",
                (camera_id, reading, threshold, severity))
    cur.execute("UPDATE camera_status SET total_anomalies = total_anomalies + 1 WHERE camera_id = ?",
                (camera_id,))
    conn.commit(); conn.close()


def upsert_camera_status(path: str, camera_id: str, avg_reading: float, is_active: int = 1):
    conn = sqlite3.connect(path); cur = conn.cursor()
    cur.execute("""
    INSERT INTO camera_status(camera_id, is_active, last_seen, total_anomalies, avg_reading)
    VALUES(?, ?, CURRENT_TIMESTAMP, COALESCE((SELECT total_anomalies FROM camera_status WHERE camera_id = ?), 0), ?)
    ON CONFLICT(camera_id) DO UPDATE SET
      is_active=excluded.is_active,
      last_seen=CURRENT_TIMESTAMP,
      avg_reading=excluded.avg_reading
    """, (camera_id, is_active, camera_id, avg_reading))
    conn.commit(); conn.close()


def recent_anomalies(path: str, limit: int = 10):
    conn = sqlite3.connect(path); cur = conn.cursor()
    cur.execute("SELECT id, camera_id, timestamp, reading, threshold, severity FROM anomalies ORDER BY timestamp DESC LIMIT ?", (limit,))
    rows = [
        {"id": r[0], "camera_id": r[1], "timestamp": r[2], "reading": r[3], "threshold": r[4], "severity": r[5]}
        for r in cur.fetchall()
    ]
    conn.close(); return rows


def camera_status_all(path: str):
    conn = sqlite3.connect(path); cur = conn.cursor()
    cur.execute("SELECT camera_id, is_active, last_seen, total_anomalies, avg_reading FROM camera_status")
    rows = [
        {"camera_id": r[0], "is_active": bool(r[1]), "last_seen": r[2], "total_anomalies": r[3], "avg_reading": r[4]}
        for r in cur.fetchall()
    ]
    conn.close(); return rows


def get_readings(path: str, camera_id: str, since):
    conn = sqlite3.connect(path); cur = conn.cursor()
    cur.execute("SELECT timestamp, reading, is_anomaly FROM dial_readings WHERE camera_id=? AND timestamp > ? ORDER BY timestamp ASC",
                (camera_id, since))
    rows = [{"timestamp": r[0], "reading": r[1], "is_anomaly": bool(r[2])} for r in cur.fetchall()]
    conn.close(); return rows

def get_last_reading_row(path: str, camera_id: str):
    conn = sqlite3.connect(path); cur = conn.cursor()
    cur.execute("SELECT timestamp, reading FROM dial_readings WHERE camera_id=? ORDER BY timestamp DESC LIMIT 1", (camera_id,))
    row = cur.fetchone()
    conn.close()
    if not row:
        return None
    return {"timestamp": row[0], "reading": float(row[1])}


def get_camera_thresholds(path: str, camera_id: str, default_low: float, default_med: float, default_high: float):
    conn = sqlite3.connect(path); cur = conn.cursor()
    cur.execute("SELECT low, med, high FROM camera_thresholds WHERE camera_id=?", (camera_id,))
    row = cur.fetchone()
    conn.close()
    if row:
        return float(row[0]), float(row[1]), float(row[2])
    return float(default_low), float(default_med), float(default_high)


def set_camera_thresholds(path: str, camera_id: str, low: float, med: float, high: float):
    conn = sqlite3.connect(path); cur = conn.cursor()
    cur.execute("""
    INSERT INTO camera_thresholds(camera_id, low, med, high) VALUES(?,?,?,?)
    ON CONFLICT(camera_id) DO UPDATE SET low=excluded.low, med=excluded.med, high=excluded.high
    """, (camera_id, low, med, high))
    conn.commit(); conn.close()


def calc_metrics(path: str):
    conn = sqlite3.connect(path); cur = conn.cursor()
    cur.execute("SELECT COUNT(*), AVG(reading), MAX(reading), MIN(reading) FROM dial_readings WHERE timestamp > datetime('now','-24 hours')")
    total, avg, mx, mn = cur.fetchone()
    total = total or 0; avg = avg or 0; mx = mx or 0; mn = mn or 0
    cur.execute("SELECT COUNT(*) FROM anomalies WHERE timestamp > datetime('now','-24 hours')")
    anoms = cur.fetchone()[0] or 0
    cur.execute("SELECT COUNT(*) FROM camera_status WHERE is_active = 1")
    active = cur.fetchone()[0] or 0
    conn.close()
    rate = round((anoms/total*100), 2) if total else 0.0
    return {
        "total_readings": int(total),
        "avg_reading": round(avg, 2),
        "max_reading": round(mx, 2),
        "min_reading": round(mn, 2),
        "anomaly_count": int(anoms),
        "anomaly_rate": rate,
        "active_cameras": int(active),
    }
