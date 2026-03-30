import cv2
import numpy as np
import time
import asyncio
import zipfile
import io
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Any
from ultralytics import YOLO
from starlette.concurrency import run_in_threadpool

from database import initialize_database, log_mission

# ==========================================
# 1. INITIALIZE & CONFIGURE MICROSERVICE
# ==========================================
app = FastAPI(
    title="Infravision Structural AI Nexus",
    description="High-performance, async-native microservice for distributed edge-AI structural damage inference.",
    version="3.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# 2. DEFINING STRICT DATA SCHEMAS
# ==========================================
class DetectionModel(BaseModel):
    class_id: int
    classification: str
    confidence: float
    bbox_xywh: List[int]

class AnalysisResponse(BaseModel):
    status: str
    engine: str
    inference_ms: float
    total_anomalies: int
    detections: List[DetectionModel]
    log_id: int = -1

class BatchAnalysisResponse(BaseModel):
    status: str
    engine: str
    total_images_processed: int
    total_inference_ms: float
    total_anomalies_detected: int
    results: Dict[str, AnalysisResponse]

# ==========================================
# 3. BACKGROUND TASKS & UTILS
# ==========================================
yolo_model = None
def load_yolo():
    global yolo_model
    if yolo_model is None:
        try:
            yolo_model = YOLO("models/exported/yolo_edge.onnx") 
        except Exception as e:
            yolo_model = YOLO("yolov8n.pt")

@app.on_event("startup")
async def startup_event():
    print("🚀 Booting Asynchronous Infravision Inference Core...")
    initialize_database()
    print("🗄️ Persistent SQLite Database Initialized.")
    await run_in_threadpool(load_yolo)
    print("✅ Core systems online and routing optimized.")

async def simulate_critical_webhook_alert(source_name: str, anomalies: int, max_conf: float):
    """Simulates a webhook/email being triggered if damage is catastrophic."""
    print(f"\n[🚨 WEBHOOK TRIGGERED] Critical Spalling detected in '{source_name}'!")
    print(f"   -> Found {anomalies} anomalies. Max severity confidence: {max_conf*100:.1f}%")
    print("   -> Dispatching automated SMS overlay to engineering field unit...")
    await asyncio.sleep(1.0) # Simulate network dispatch
    print("[🚨 WEBHOOK TRIGGERED] Dispatch complete.\n")

def process_detections_and_log(detections, source_name, engine_name, bg_tasks: BackgroundTasks):
    """Helper to log the mission to DB and potentially trigger an alert."""
    log_id = log_mission(source_name=source_name, engine=engine_name, total_anomalies=len(detections), list_of_detections=detections) if log_mission else -1
    
    # Alert Logic
    critical_count = sum(1 for d in detections if getattr(d, 'class_id', 0) == 2 and getattr(d, 'confidence', 0) > 0.90)
    if critical_count > 0:
        max_conf = max(getattr(d, 'confidence', 0) for d in detections)
        bg_tasks.add_task(simulate_critical_webhook_alert, source_name, len(detections), max_conf)
        
    return log_id

# ==========================================
# 4. CPU-BOUND INFERENCE LOGIC (Threaded)
# ==========================================
def cpu_bound_heuristic(image_bytes, thresh_conf):
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None: raise ValueError("Unreadable image matrix.")
        
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    edges = cv2.Canny(blurred, 30, 100)
    kernel = np.ones((5,5), np.uint8)
    edges_dilated = cv2.dilate(edges, kernel, iterations=1)
    contours, _ = cv2.findContours(edges_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    detections = []
    for c in contours:
        area = cv2.contourArea(c)
        if area > 120: 
            x, y, w, h = cv2.boundingRect(c)
            if x == 0 or y == 0 or x+w >= image.shape[1]-5 or y+h >= image_rgb.shape[0]-5: continue
            
            conf = min(0.99, area / 1200.0) + np.random.uniform(0.01, 0.05)
            conf = min(0.99, conf)
            
            if conf >= thresh_conf:
                aspect_ratio = float(w)/h if h > 0 else 1
                if aspect_ratio > 2.2 or aspect_ratio < 0.45:
                    cls_id = 1; label = "Crack"
                else:
                    cls_id = 2; label = "Spall"
                
                detections.append(DetectionModel(
                    class_id=cls_id,
                    classification=label,
                    confidence=round(float(conf), 3),
                    bbox_xywh=[int(x), int(y), int(w), int(h)]
                ))
    return detections

def cpu_bound_yolo(image_bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None: raise ValueError("Unreadable image matrix.")
        
    results = yolo_model(image)[0]
    
    detections = []
    for box in results.boxes:
        r = box.xyxy[0].tolist()
        c = int(box.cls[0].item())
        conf = box.conf[0].item()
        
        names = getattr(yolo_model, 'names', {0: 'Intact', 1: 'Crack', 2: 'Spall'})
        label = names.get(c, "Unknown")
            
        x1, y1, x2, y2 = r
        w, h = x2 - x1, y2 - y1
        
        detections.append(DetectionModel(
            class_id=c,
            classification=label,
            confidence=round(float(conf), 3),
            bbox_xywh=[int(x1), int(y1), int(w), int(h)]
        ))
    return detections

# ==========================================
# 5. ASYNCHRONOUS API ENDPOINTS
# ==========================================
@app.get("/health")
async def system_health():
    return {"status": "operational", "model_loaded": yolo_model is not None}

@app.post("/analyze/heuristic", response_model=AnalysisResponse)
async def analyze_heuristic(bg_tasks: BackgroundTasks, file: UploadFile = File(...), confidence_threshold: float = Form(0.30)):
    start_time = time.time()
    try:
        contents = await file.read()
        detections = await run_in_threadpool(cpu_bound_heuristic, contents, confidence_threshold)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image processing failed: {str(e)}")
        
    inference_time = (time.time() - start_time) * 1000.0
    
    # Database and Alert Checks
    engine_name = "heuristic_opencv"
    log_id = log_mission(file.filename, engine_name, len(detections), detections)
    
    critical_count = sum(1 for d in detections if d.class_id == 2 and d.confidence > 0.90)
    if critical_count > 0:
        max_conf = max(d.confidence for d in detections)
        bg_tasks.add_task(simulate_critical_webhook_alert, file.filename, len(detections), max_conf)
    
    return AnalysisResponse(
        status="success", engine=engine_name, inference_ms=round(inference_time, 2),
        total_anomalies=len(detections), detections=detections, log_id=log_id
    )

@app.post("/analyze/yolo", response_model=AnalysisResponse)
async def analyze_yolo_dl(bg_tasks: BackgroundTasks, file: UploadFile = File(...)):
    start_time = time.time()
    try:
        contents = await file.read()
        detections = await run_in_threadpool(cpu_bound_yolo, contents)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Deep Learning inference failed: {str(e)}")
        
    inference_time = (time.time() - start_time) * 1000.0
    engine_name = "yolo_v8_onnx"
    
    log_id = log_mission(file.filename, engine_name, len(detections), detections)
    critical_count = sum(1 for d in detections if d.class_id == 2 and d.confidence > 0.90)
    if critical_count > 0:
        max_conf = max(d.confidence for d in detections) if detections else 0
        bg_tasks.add_task(simulate_critical_webhook_alert, file.filename, len(detections), max_conf)
    
    return AnalysisResponse(
        status="success", engine=engine_name, inference_ms=round(inference_time, 2),
        total_anomalies=len(detections), detections=detections, log_id=log_id
    )

@app.post("/analyze/batch", response_model=BatchAnalysisResponse)
async def analyze_batch_zip(bg_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """Accepts a .zip file of images, unzips in memory, and maps the whole dataset."""
    if not file.filename.endswith('.zip'):
        raise HTTPException(status_code=400, detail="Must upload a .zip file containing images.")
        
    start_time = time.time()
    contents = await file.read()
    results_map = {}
    total_anomalies = 0
    
    try:
        # Process Zip entirely in memory using a virtual buffer
        with zipfile.ZipFile(io.BytesIO(contents)) as zip_ref:
            for file_name in zip_ref.namelist():
                if file_name.endswith(('.png', '.jpg', '.jpeg')):
                    img_bytes = zip_ref.read(file_name)
                    # We'll batch to YOLO engine for max accuracy
                    detections = await run_in_threadpool(cpu_bound_yolo, img_bytes)
                    
                    log_id = log_mission(f"BATCH: {file_name}", "yolo_v8_onnx", len(detections), detections)
                    total_anomalies += len(detections)
                    
                    results_map[file_name] = AnalysisResponse(
                        status="success", engine="yolo_v8_onnx", inference_ms=0.0, 
                        total_anomalies=len(detections), detections=detections, log_id=log_id
                    )
    except Exception as e:
         raise HTTPException(status_code=500, detail=f"Failed to process zip archive: {str(e)}")
         
    total_inference_ms = (time.time() - start_time) * 1000.0
    
    return BatchAnalysisResponse(
        status="success",
        engine="yolo_v8_onnx_batch",
        total_images_processed=len(results_map),
        total_inference_ms=round(total_inference_ms, 2),
        total_anomalies_detected=total_anomalies,
        results=results_map
    )
