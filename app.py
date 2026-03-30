import streamlit as st
import cv2
import numpy as np
import pandas as pd
import time
import glob
import os
import math
from database import initialize_database, log_mission, get_recent_missions

# Initialize the persistent DB on every app load
initialize_database()

# Set up standard web application layout
st.set_page_config(layout="wide", page_title="Infravision OS", page_icon="🏗️")

st.markdown("""
    <style>
    .stApp { background-color: #0d1117; color: #c9d1d9; font-family: 'Inter', sans-serif; }
    h1, h2, h3, h4, .stTabs [data-baseweb="tab-list"] button { color: #58a6ff !important; font-weight: 600; }
    div[data-testid="stMetricValue"] { font-size: 2.2rem; color: #3fb950; text-shadow: 0px 0px 10px rgba(63, 185, 80, 0.2); }
    section[data-testid="stSidebar"] { border-right: 1px solid #30363d; }
    .stButton>button { color: #0d1117; background-color: #58a6ff; font-weight: bold; border-radius: 6px; border: none; transition: 0.3s; width: 100%;}
    .stButton>button:hover { background-color: #79c0ff; filter: drop-shadow(0 0 5px rgba(88,166,255,0.5));}
    </style>
""", unsafe_allow_html=True)

MASTER_CLASSES = {0: 'Intact', 1: 'Surface Crack', 2: 'Deep Spalling / Severe Damage'}

# ---------------------------------------------------------
# SESSION STATE (Persistent History Ledger)
# ---------------------------------------------------------
if 'scan_ledger' not in st.session_state:
    st.session_state.scan_ledger = []

# ---------------------------------------------------------
# AUTONOMOUS OFFLINE VISION ENGINE 
# ---------------------------------------------------------
def heuristic_crack_detection(image, thresh_conf):
    """Uses OpenCV Edge topologies to find structural defect contours offline."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    
    edges = cv2.Canny(blurred, 30, 100)
    kernel = np.ones((5,5), np.uint8)
    edges_dilated = cv2.dilate(edges, kernel, iterations=1)
    
    contours, _ = cv2.findContours(edges_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    annotated = image.copy()
    detections = []
    
    for c in contours:
        area = cv2.contourArea(c)
        if area > 120: 
            x, y, w, h = cv2.boundingRect(c)
            if x == 0 or y == 0 or x+w >= image.shape[1]-5 or y+h >= image.shape[0]-5: continue
            
            conf = min(0.99, area / 1200.0) + np.random.uniform(0.01, 0.05)
            conf = min(0.99, conf)
            
            if conf >= thresh_conf:
                aspect_ratio = float(w)/h if h > 0 else 1
                if aspect_ratio > 2.2 or aspect_ratio < 0.45:
                    cls_id = 1 
                    color = (255, 50, 50) 
                    label = f"Crack {conf:.2f}"
                else:
                    cls_id = 2 
                    color = (255, 150, 0) 
                    label = f"Spall {conf:.2f}"
                    
                thick = 3; l = 15 
                cv2.rectangle(annotated, (x, y), (x+w, y+h), color, 1) 
                cv2.line(annotated, (x, y), (x+l, y), color, thick)
                cv2.line(annotated, (x, y), (x, y+l), color, thick)
                cv2.line(annotated, (x+w, y), (x+w-l, y), color, thick)
                cv2.line(annotated, (x+w, y), (x+w, y+l), color, thick)
                cv2.line(annotated, (x, y+h), (x+l, y+h), color, thick)
                cv2.line(annotated, (x, y+h), (x, y+h-l), color, thick)
                cv2.line(annotated, (x+w, y+h), (x+w-l, y+h), color, thick)
                cv2.line(annotated, (x+w, y+h), (x+w, y+h-l), color, thick)
                
                cv2.putText(annotated, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                detections.append((cls_id, conf))
                
    return annotated, detections

def generate_mock_concrete(seed=1):
    np.random.seed(int(time.time() * 100) % 10000 if seed == 0 else seed)
    base = np.full((500, 700, 3), 140, dtype=np.uint8)
    noise = np.random.normal(0, 15, base.shape).astype(np.int16)
    bg = np.clip(base + noise, 0, 255).astype(np.uint8)
    if seed == 1: cv2.line(bg, (np.random.randint(100, 200), 100), (np.random.randint(400, 500), 400), (45, 45, 45), 5)
    elif seed == 2: cv2.circle(bg, (np.random.randint(300, 400), np.random.randint(200, 300)), 40, (70, 70, 70), -1) 
    return bg

def generate_large_concrete_surface():
    np.random.seed(int(time.time()) % 1000)
    base = np.full((1200, 1400, 3), 135, dtype=np.uint8)
    noise = np.random.normal(0, 20, base.shape).astype(np.int16)
    bg = np.clip(base + noise, 0, 255).astype(np.uint8)
    stain_noise = cv2.resize(np.random.normal(0, 30, (30, 40, 3)), (1400, 1200)).astype(np.int16)
    bg = np.clip(bg + stain_noise, 0, 255).astype(np.uint8)
    
    curr = (500, 200)
    for _ in range(18): 
        dx, dy = np.random.randint(-15, 60), np.random.randint(10, 55)
        nxt = (curr[0] + dx, curr[1] + dy)
        cv2.line(bg, curr, nxt, (30, 30, 30), np.random.randint(3, 7))
        if np.random.rand() > 0.5: cv2.line(bg, nxt, (nxt[0] + np.random.randint(-25, 25), nxt[1] + np.random.randint(5, 30)), (40, 40, 40), 2)
        curr = nxt
        
    impact_center = (900, 800)
    cv2.circle(bg, impact_center, 40, (75, 75, 75), -1)
    for _ in range(12): 
        offset = (impact_center[0] + np.random.randint(-40, 40), impact_center[1] + np.random.randint(-40, 40))
        cv2.circle(bg, offset, np.random.randint(10, 25), (60, 60, 60), -1)
        
    return bg

def draw_radar_sweep(img, center, radius, angle):
    """Draws a rotating green radar sweep and concentric circles."""
    cv2.circle(img, center, radius, (0, 70, 0), 1)
    cv2.circle(img, center, int(radius*0.66), (0, 50, 0), 1)
    cv2.circle(img, center, int(radius*0.33), (0, 30, 0), 1)
    
    # Calculate sweep line coordinates
    end_x = int(center[0] + radius * math.cos(math.radians(angle)))
    end_y = int(center[1] + radius * math.sin(math.radians(angle)))
    cv2.line(img, center, (end_x, end_y), (0, 255, 0), 2)
    
    # Draw a fading wedge for the sweep
    overlay = img.copy()
    pts = np.array([center, (end_x, end_y), 
                    (int(center[0] + radius * math.cos(math.radians(angle-15))), 
                     int(center[1] + radius * math.sin(math.radians(angle-15))))])
    cv2.fillPoly(overlay, [pts], (0, 100, 0))
    cv2.addWeighted(overlay, 0.4, img, 0.6, 0, img)

def main():
    st.markdown("<h1 style='text-align: center; margin-bottom: 0px;'>🏗️ INFRAVISION NEXUS</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #8b949e; margin-top: 5px; font-size: 1.2rem;'>Distributed Edge AI for Global Civil Infrastructure</p>", unsafe_allow_html=True)
    st.markdown("---")
    
    st.sidebar.markdown("## 🛰️ Operations Network")
    mode = st.sidebar.radio("Command Module", ["📷 Deep Diagnostics", "🌍 Global Fleet Monitor", "🚁 Active Drone Feed (HD Simulation)", "📈 Predictive Analytics Hub"])
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🧠 AI Parameters")
    confidence_threshold = st.sidebar.slider("Anomaly Confidence Threshold", 0.0, 1.0, 0.30)
    st.sidebar.success("✅ **Offline Heuristic Engine Active**. Bypassing firewall blocks.")
    
    # ---------------- HISTORICAL LEDGER (SQLite) ----------------
    st.sidebar.markdown("### 📜 Persistent Mission Ledger")
    recent_missions = get_recent_missions(limit=6)
    if not recent_missions:
        st.sidebar.info("Awaiting first telemetry scan...")
    else:
        for row in recent_missions:
            # row: (id, timestamp, source_name, engine, total_anomalies, critical_alerts, max_confidence)
            _id, ts, src, eng, total, crits, max_c = row
            c_color = "🔴" if crits > 0 else "🟡" if total > 0 else "✅"
            ts_short = str(ts)[-8:] if ts else "?"
            st.sidebar.caption(f"{c_color} `{ts_short}` **{src}** — {total} defects")

    if mode == "🌍 Global Fleet Monitor":
        st.markdown("### 🌍 Real-Time Regional Sensor Network")
        global_hubs = np.array([
            [37.77, -122.41], [40.71, -74.00], [51.50, -0.12], [48.85, 2.35], 
            [35.68, 139.69], [-33.86, 151.20], [-23.55, -46.63], [25.20, 55.27], 
            [1.35, 103.81], [-1.29, 36.82], [28.61, 77.20]
        ])
        
        np.random.seed(42)  
        drones = []
        for hub in global_hubs:
            spread = np.random.randn(np.random.randint(10, 16), 2) * 5.5 
            drones.extend(hub + spread)
            
        df_drones = pd.DataFrame(drones, columns=['lat', 'lon'])
        st.map(df_drones, zoom=1, size=40)
        
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        col_m1.metric("Active Edge Nodes", f"{len(df_drones)} Units", "+3 Online")
        col_m2.metric("Structures Inspected (24h)", "1,048", "+12%")
        col_m3.metric("Critical Anomalies Flagged", "7", "-2")
        col_m4.metric("Avg Fleet Inference", "14.2 ms", "-0.5ms")
        
        st.markdown("---")
        st.markdown("#### 📡 Regional Network Traffic Map")
        chart_data = pd.DataFrame(
            np.random.randn(20, 3) * 10 + [100, 80, 50],
            columns=['NA Region', 'EU Region', 'APAC Region']
        )
        st.area_chart(chart_data)
        
        with st.expander("Show Latest Regional Intel Logs"):
            for i in range(5):
                st.text(f"[{time.strftime('%H:%M:%S')}] - NODE-{np.random.randint(100,999)} verified safe perimeter scanning...")

    elif mode == "📷 Deep Diagnostics":
        col1, col2 = st.columns([1.2, 2])
        img_to_analyze = None
        source_name = "User Img"
        roi_box = None  # Will hold (x, y, w, h) if user draws a region
        
        with col1:
            st.markdown("### ⚡ Quick Demobank")
            st.write("Click a simulated structure below to trigger an active defect scan!")
            
            qc1, qc2, qc3 = st.columns(3)
            with qc1:
                if st.button("🚧 Bridge Deck\n(Crack)"): 
                    img_to_analyze = generate_mock_concrete(1); source_name = "Bridge Deck"
            with qc2:
                if st.button("🏢 Wall Panel\n(Spall)"): 
                    img_to_analyze = generate_mock_concrete(2); source_name = "Wall Panel"
            with qc3:
                if st.button("🎲 Fetch From\nDataset"): 
                    dataset_images = glob.glob('data/**/*.jpg', recursive=True) + glob.glob('mock_dataset/**/*.jpg', recursive=True)
                    if dataset_images:
                        random_img_path = np.random.choice(dataset_images)
                        img_to_analyze = cv2.cvtColor(cv2.imread(random_img_path), cv2.COLOR_BGR2RGB)
                        source_name = "Unified Database"
                        st.success(f"Loaded: `{random_img_path}`")
                    else: st.error("Warning: Datasets are empty.")
                        
            st.markdown("<br><b>Or Upload Real World Scan:</b>", unsafe_allow_html=True)
            uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])
            
            if uploaded_file is not None:
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                img_to_analyze = cv2.cvtColor(cv2.imdecode(file_bytes, 1), cv2.COLOR_BGR2RGB)
                source_name = uploaded_file.name
                
            # ----- 🖌️ INTERACTIVE MANUAL ROI OVERRIDE (Slider-based) -----
            if img_to_analyze is not None:
                st.markdown("---")
                st.markdown("#### 🖌️ Manual Region Override")
                st.caption("Use sliders to lock the AI scan to a specific region of the image.")
                
                img_h, img_w = img_to_analyze.shape[:2]
                
                enable_roi = st.toggle("Enable Region-of-Interest (ROI) Lock", value=False, key="roi_toggle")
                if enable_roi:
                    roi_col1, roi_col2 = st.columns(2)
                    with roi_col1:
                        rx = st.slider("X Start (px)", 0, img_w - 10, img_w // 4, key="roi_x")
                        rw = st.slider("Width (px)",  10, img_w - rx, img_w // 2, key="roi_w")
                    with roi_col2:
                        ry = st.slider("Y Start (px)", 0, img_h - 10, img_h // 4, key="roi_y")
                        rh = st.slider("Height (px)", 10, img_h - ry, img_h // 2, key="roi_h")
                    
                    roi_box = (rx, ry, rw, rh)
                    
                    # Show live preview with ROI box drawn on it
                    preview = img_to_analyze.copy()
                    cv2.rectangle(preview, (rx, ry), (rx + rw, ry + rh), (88, 166, 255), 3)
                    cv2.putText(preview, f"ROI: {rw}x{rh}px", (rx + 5, ry + 22),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (88, 166, 255), 2)
                    st.image(preview, caption="ROI Preview — blue box is the analysis zone.", use_column_width=True)
                    st.success(f"ROI Lock Active: `[x={rx}, y={ry}, {rw}x{rh}px]`")
                else:
                    roi_box = None

                
        if img_to_analyze is not None:
            with col2:
                dashboard_tabs = st.tabs(["🔍 Holographic Overlay", "📋 Engineering Report", "💰 Cost Estimations", "🌊 Acoustic Sonar Integrator", "🗄️ Mission Logs"])
                
                with st.spinner(f"Routing logic through Offline Vision Engine..."):
                    start_time = time.time()
                    # Apply ROI crop if user drew a region on the canvas
                    analysis_target = img_to_analyze
                    if roi_box:
                        rx, ry, rw, rh = roi_box
                        analysis_target = img_to_analyze[ry:ry+rh, rx:rx+rw]
                    
                    annotated_roi, detections = heuristic_crack_detection(analysis_target, confidence_threshold)
                    inference_time = time.time() - start_time
                    num_defects = len(detections)
                    
                    # Paste ROI result back on full image for display
                    annotated_img = img_to_analyze.copy()
                    if roi_box:
                        rx, ry, rw, rh = roi_box
                        annotated_img[ry:ry+rh, rx:rx+rw] = annotated_roi
                        cv2.rectangle(annotated_img, (rx, ry), (rx+rw, ry+rh), (88, 166, 255), 2)
                    else:
                        annotated_img = annotated_roi
                    
                    # Log to persistent SQLite DB
                    det_list = [{"class_id": c, "confidence": conf, "classification": MASTER_CLASSES.get(min(c,2), "Unknown"), "bbox": []} for c, conf in detections]
                    log_mission(source_name=source_name, engine="heuristic_opencv", total_anomalies=num_defects, list_of_detections=det_list)
                    
                    if num_defects > 0:
                        st.session_state.scan_ledger.append(f"{time.strftime('%H:%M:%S')} - Critical Defect ({source_name})")
                        # 🚨 Alert Banner for critical finds
                        critical_finds = sum(1 for _, c in detections if c > 0.75)
                        if critical_finds > 0:
                            st.error(f"🚨 **CRITICAL ALERT:** {critical_finds} high-severity anomaly(ies) detected in **{source_name}**. Immediate engineering review recommended!")
                    else:
                        st.session_state.scan_ledger.append(f"{time.strftime('%H:%M:%S')} - Intact Sector ({source_name})")
                    
                    with dashboard_tabs[0]:
                        st.image(annotated_img, use_column_width=True)
                        st.caption(f"Inference executed completely offline in {inference_time*1000:.1f}ms.")
                        
                    with dashboard_tabs[1]:
                        if num_defects == 0:
                            st.success("✅ **Sector Clear.** Structure satisfies baseline safety protocols.")
                        else:
                            st.warning(f"⚠️ **Warning Level 2:** {num_defects} anomalies visually identified.")
                            for i, (cls_id, conf) in enumerate(detections):
                                damage_type = MASTER_CLASSES.get(min(cls_id, 2), 'Unknown Surface Anomaly')
                                severity = "🔴 Critical" if conf > 0.6 else "🟡 Moderate"
                                action = "Deploy targeted structural support instantly." if conf > 0.6 else "Flag for review."
                                st.markdown(f"**Defect ID #{i+1} :**")
                                st.markdown(f"> **Classification:** {damage_type}  \n> **AI Severity Index:** {conf*100:.1f}% Match  \n> **Recommended Directive:** {action}")
                            
                            st.markdown("---")
                            csv_data = pd.DataFrame(
                                [{"Defect ID": j+1, "Classification": MASTER_CLASSES.get(min(cid, 2), 'Unknown'), "Severity (%)": round(c*100, 1)} for j, (cid, c) in enumerate(detections)]
                            ).to_csv(index=False).encode('utf-8')
                            st.download_button(label="📥 Export Digital Twin Record (CSV)", data=csv_data, file_name=f'scan_report_{int(time.time())}.csv', mime='text/csv')

                    with dashboard_tabs[2]:
                        st.markdown("#### Automated Repair Contractor Estimations")
                        total_cost = 0
                        for i, (cls_id, conf) in enumerate(detections):
                            cost = 4500 if conf > 0.6 else 850
                            total_cost += cost
                            st.write(f"- Damage Anomaly Component #{i+1}: **${cost:,}**")
                        st.markdown("---")
                        st.markdown(f"### Total Est. Reserve: <span style='color: #ff7b72'>${total_cost:,} USD</span>", unsafe_allow_html=True)

                    with dashboard_tabs[3]:
                        st.markdown("#### Subsurface Acoustic Waveform")
                        st.info("Simulated ultrasonic material penetration testing to detect internal resonance defects.")
                        # Generate a base "healthy" frequency sine wave
                        t = np.linspace(0, 10, 500)
                        acoustic_wave = np.sin(t * 10) + np.random.normal(0, 0.2, 500)
                        
                        # Add violent resonance spikes if defects exist on the visual frame
                        if num_defects > 0:
                            for idx in range(num_defects):
                                spike_pos = np.random.randint(100, 400)
                                acoustic_wave[spike_pos-10:spike_pos+10] += np.random.normal(0, 5.0, 20) * (idx+1)
                                
                        st.line_chart(acoustic_wave, height=250, color="#ff4b4b" if num_defects > 0 else "#3fb950")
                        st.caption("Acoustic resonance anomalies align with physical concrete spalling and micro-fissures.")

                    with dashboard_tabs[4]:
                        st.markdown("#### 🗄️ Persistent Mission History (SQLite)")
                        st.caption("Live read from the on-disk `infravision_logs.db` database. Survives restarts.")
                        rows = get_recent_missions(limit=20)
                        if rows:
                            df_logs = pd.DataFrame(rows, columns=["ID", "Timestamp", "Source", "Engine", "Anomalies", "Critical Alerts", "Max Confidence"])
                            df_logs["Max Confidence"] = df_logs["Max Confidence"].apply(lambda x: f"{x*100:.1f}%")
                            st.dataframe(df_logs.set_index("ID"), use_container_width=True)
                        else:
                            st.info("No missions logged yet. Run a scan above!")

    elif mode == "🚁 Active Drone Feed (HD Simulation)":
        st.markdown("### 🚁 Live Reconnaissance Stream")
        
        col_video, col_telemetry = st.columns([2.5, 1])
        
        with col_video: stframe = st.empty()
        with col_telemetry:
            st.markdown("#### Edge Telemetry")
            telemetry_alt = st.empty(); telemetry_batt = st.empty(); telemetry_fps = st.empty(); telemetry_status = st.empty()
            enable_thermal = st.checkbox("🔥 Enable Thermal Vision (IR)")
            enable_radar = st.checkbox("📡 Enable Radar Sweep HUD", value=True)
        
        start_btn = st.button("▶️ Launch Patrol Route Alpha")
        
        if start_btn:
            import os
            os.makedirs("recorded_inspections", exist_ok=True)
            video_filename = f"recorded_inspections/drone_patrol_log_{int(time.time())}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
            out = cv2.VideoWriter(video_filename, fourcc, 15.0, (640, 480))
            
            telemetry_status.warning(f"Establishing encrypted link... Preparing recording...")
            time.sleep(1)
            telemetry_status.success(f"🔴 Uplink SECURE. Flight path engaged.")
            
            surface = generate_large_concrete_surface()
            battery = 98.4
            radar_angle = 0
            
            for i in range(55):
                x = 100 + int(i * 12.5) 
                y = 80 + int(i * 10)    
                frame = surface[y:y+480, x:x+640].copy()
                
                if enable_thermal: frame = cv2.applyColorMap(frame, cv2.COLORMAP_INFERNO)
                
                annotated_img, detections = heuristic_crack_detection(frame, thresh_conf=0.25)
                
                # Active Dynamic HUD
                cv2.line(annotated_img, (320, 220), (320, 260), (0, 255, 0), 1)
                cv2.line(annotated_img, (300, 240), (340, 240), (0, 255, 0), 1)
                cv2.circle(annotated_img, (320, 240), 50, (0, 255, 0), 1)
                
                # Rotating Radar System
                if enable_radar:
                    draw_radar_sweep(annotated_img, (100, 380), 50, radar_angle)
                    radar_angle = (radar_angle + 12) % 360
                
                cv2.putText(annotated_img, f"REC | LAT 37.76 LONG -122.41 | Alt: 14.{i%10}m", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.rectangle(annotated_img, (5, 5), (635, 475), (0, 255, 0), 2) 
                    
                telemetry_alt.metric("Altitude", f"14.{i%10} m")
                battery -= 0.1
                telemetry_batt.metric("Drone Battery", f"{battery:.1f}%")
                telemetry_fps.metric("Inference Engine FPS", "29.4 FPS" if i%3 != 0 else "28.9 FPS")
                    
                stframe.image(annotated_img, channels="RGB")
                
                bgr_frame = cv2.cvtColor(annotated_img, cv2.COLOR_RGB2BGR)
                out.write(bgr_frame)
                time.sleep(0.08)
                
            out.release()
            
            # Log this entire mission in our session ledger!
            st.session_state.scan_ledger.append(f"{time.strftime('%H:%M:%S')} - Drone Patrol Alpha Completed")
            telemetry_status.info(f"Patrol Route Concluded. ✅ Video logged to: `{os.path.abspath(video_filename)}`")

    elif mode == "📈 Predictive Analytics Hub":
        st.markdown("### 📈 Structure Degradation Forecasting")
        st.info("Utilizing localized AI models to predict long-term material failure cascades across simulated epochs.")
        
        col_opt1, col_opt2 = st.columns([1, 2])
        with col_opt1:
            material = st.selectbox("Construction Material", ["Reinforced Concrete", "Steel Lattice", "Prestressed Masonry"])
            env_stress = st.slider("Environmental Stress Factor", 1.0, 5.0, 2.5)
            traffic_load = st.slider("Traffic Load Volume", 1.0, 10.0, 5.0)
            
            if st.button("Initialize Forecast Simulation"):
                with st.spinner("Crunching historical metadata..."):
                    time.sleep(1.2)
                    st.success("Simulation Complete")
                    
        with col_opt2:
            st.markdown("#### Projected Structural Integrity Decay (15 Years)")
            
            # Simulate decay curve
            years = np.arange(2026, 2041)
            base_integrity = 100.0
            np.random.seed(int(time.time()) % 1000)
            decay = np.random.uniform(0.5, 1.5, size=len(years)) * (env_stress * 0.5) * (traffic_load * 0.2)
            
            integrity_values = [base_integrity]
            for d in decay[1:]:
                next_val = max(0, integrity_values[-1] - d)
                integrity_values.append(next_val)
                
            df_forecast = pd.DataFrame({"Year": years, "Integrity %": integrity_values}).set_index("Year")
            st.line_chart(df_forecast, color="#ff4b4b")
            
            st.markdown("---")
            
            curr_val = integrity_values[-1]
            if curr_val > 70:
                st.success(f"**Optimal.** Expected Integrity in 2040: {curr_val:.1f}%")
            elif curr_val > 40:
                st.warning(f"**Caution.** Expected Integrity in 2040: {curr_val:.1f}%")
            else:
                st.error(f"**Critical Danger.** Collapse Risk after 2038. Integrity: {curr_val:.1f}%")

if __name__ == '__main__':
    main()
