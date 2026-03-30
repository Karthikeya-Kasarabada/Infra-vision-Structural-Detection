import sqlite3
import time
from typing import List, Dict

DB_PATH = "infravision_logs.db"

def initialize_database():
    """Sets up the SQLite database and logs table if they don't exist."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Mission Logs Table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS mission_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            source_name TEXT,
            engine TEXT,
            total_anomalies INTEGER,
            critical_alerts INTEGER,
            max_confidence REAL
        )
    ''')
    
    # Detailed Detections Table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            log_id INTEGER,
            class_id INTEGER,
            classification TEXT,
            confidence REAL,
            bbox TEXT,
            FOREIGN KEY(log_id) REFERENCES mission_logs(id)
        )
    ''')
    
    conn.commit()
    conn.close()

def log_mission(source_name: str, engine: str, total_anomalies: int, list_of_detections: List):
    """Logs a completed inference mission and returns the log_id."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    critical_count = sum(1 for d in list_of_detections if getattr(d, 'confidence', d.get('confidence', 0)) > 0.8)
    max_conf = max([getattr(d, 'confidence', d.get('confidence', 0)) for d in list_of_detections]) if list_of_detections else 0.0
    
    cursor.execute('''
        INSERT INTO mission_logs (source_name, engine, total_anomalies, critical_alerts, max_confidence)
        VALUES (?, ?, ?, ?, ?)
    ''', (source_name, engine, total_anomalies, critical_count, max_conf))
    
    log_id = cursor.lastrowid
    
    for d in list_of_detections:
        cls_id = getattr(d, 'class_id', d.get('class_id', -1))
        classification = getattr(d, 'classification', d.get('classification', 'Unknown'))
        conf = getattr(d, 'confidence', d.get('confidence', 0.0))
        bbox = str(getattr(d, 'bbox_xywh', d.get('bbox', [])))
        
        cursor.execute('''
            INSERT INTO detections (log_id, class_id, classification, confidence, bbox)
            VALUES (?, ?, ?, ?, ?)
        ''', (log_id, cls_id, classification, conf, bbox))
        
    conn.commit()
    conn.close()
    return log_id

def get_recent_missions(limit=10):
    """Retrieves the most recent inference missions."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM mission_logs ORDER BY timestamp DESC LIMIT ?', (limit,))
    rows = cursor.fetchall()
    conn.close()
    return rows
