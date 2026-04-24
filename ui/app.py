import streamlit as st
import os
import json
import cv2
import numpy as np
from pathlib import Path
import sys
import requests
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Add engine to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from engine.dsg_core import DynamicSceneGraph
from components import (
    render_match_dashboard,
    render_video_player,
    render_event_stream,
    render_llm_commentary,
    render_agent_outputs,
    render_match_controls
)
from agent_integration import LLMAgentPool
from config import DOMAINS

# Page configuration
st.set_page_config(
    page_title="FusionX: Real-Time Event Engine",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .event-alert {
        background-color: #fff3cd;
        padding: 12px;
        border-radius: 5px;
        border-left: 5px solid #ffc107;
        margin: 8px 0;
    }
    .commentary-box {
        background-color: #f0f4f8;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 10px 0;
        font-style: italic;
    }
    .agent-output {
        background-color: #e8f5e9;
        padding: 12px;
        border-radius: 5px;
        margin: 8px 0;
        border-left: 4px solid #4caf50;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
def init_session_state():
    if 'selected_domain' not in st.session_state:
        st.session_state.selected_domain = "Cricket"
    
    domain_cfg = DOMAINS[st.session_state.selected_domain]
    
    if 'domain_state' not in st.session_state:
        st.session_state.domain_state = domain_cfg['default_state'].copy()
    
    if 'events_log' not in st.session_state:
        st.session_state.events_log = []
    
    if 'commentary_history' not in st.session_state:
        st.session_state.commentary_history = []
    
    if 'dsg_engine' not in st.session_state:
        config_path = os.path.join(os.path.dirname(__file__), '..', 'engine', 'config.json')
        if os.path.exists(config_path):
            st.session_state.dsg_engine = DynamicSceneGraph(config_path)
        else:
            st.session_state.dsg_engine = None
    
    if 'agent_pool' not in st.session_state:
        st.session_state.agent_pool = LLMAgentPool()
    
    if 'playing' not in st.session_state:
        st.session_state.playing = False
    
    if 'current_frame' not in st.session_state:
        st.session_state.current_frame = 0
    
    if 'selected_video' not in st.session_state:
        st.session_state.selected_video = None

init_session_state()

# Get available videos from assets folder
def get_available_videos():
    assets_path = os.path.join(os.path.dirname(__file__), '..', 'assets')
    if not os.path.exists(assets_path):
        os.makedirs(assets_path)
    
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv'}
    videos = []
    
    for file in os.listdir(assets_path):
        if os.path.splitext(file)[1].lower() in video_extensions:
            videos.append(file)
    
    return sorted(videos)

# Main layout
domain_cfg = DOMAINS[st.session_state.selected_domain]
st.title(f"{domain_cfg['icon']} FusionX: {st.session_state.selected_domain} Event Engine")

# Sidebar
with st.sidebar:
    st.header("Configuration")
    
    # Domain Selection
    st.subheader("Domain Selection")
    selected_domain = st.selectbox(
        "Select Application Domain",
        list(DOMAINS.keys()),
        index=list(DOMAINS.keys()).index(st.session_state.selected_domain)
    )
    
    if selected_domain != st.session_state.selected_domain:
        st.session_state.selected_domain = selected_domain
        st.session_state.domain_state = DOMAINS[selected_domain]['default_state'].copy()
        st.session_state.events_log = []
        st.session_state.commentary_history = []
        
        # Load correct DSG config for domain
        config_file = 'config_traffic.json' if selected_domain == 'Traffic' else 'config.json'
        config_path = os.path.join(os.path.dirname(__file__), '..', 'engine', config_file)
        if os.path.exists(config_path):
            st.session_state.dsg_engine = DynamicSceneGraph(config_path)
            
        st.rerun()
    
    domain_cfg = DOMAINS[selected_domain]
    st.info(f"{domain_cfg['icon']} {domain_cfg['description']}")
    
    st.divider()
    
    # Video selection
    videos = get_available_videos()
    videos.insert(0, "YouTube Link (Stream)")
    videos.insert(0, "Screen Share (Live)")
    videos.insert(0, "Live Webcam (0)")
    if videos:
        selected_video = st.selectbox("Select Video", videos)
        st.session_state.selected_video = selected_video
        
        youtube_url = None
        if selected_video == "YouTube Link (Stream)":
            youtube_url = st.text_input("Enter YouTube URL (e.g., https://youtu.be/...):")
    else:
        st.warning("No videos found in assets folder")
        selected_video = None
    
    st.divider()
    
    # Vision Engine Configuration
    st.subheader("👁️ Vision Engine")
    vision_engine = st.radio(
        "Select Detection Architecture",
        ["YOLOv8 + Spatial Math", "RelTR Transformer (Phase 3 Experimental)"]
    )
    st.session_state.vision_engine = vision_engine
    
    st.divider()

    # Domain-specific configuration
    st.subheader(f"{domain_cfg['icon']} {selected_domain} Setup")
    
    if selected_domain == "Cricket":
        st.session_state.domain_state['striker'] = st.text_input("Striker", st.session_state.domain_state.get('striker', ''))
        st.session_state.domain_state['bowler'] = st.text_input("Bowler", st.session_state.domain_state.get('bowler', ''))
    elif selected_domain == "Security":
        st.session_state.domain_state['location'] = st.text_input("Camera Location", st.session_state.domain_state.get('location', ''))
    elif selected_domain == "Traffic":
        st.session_state.domain_state['intersection'] = st.text_input("Intersection Name", st.session_state.domain_state.get('intersection', ''))

    st.session_state.domain_state['phase'] = st.selectbox(
        "Operation Phase",
        domain_cfg['phases'],
        index=domain_cfg['phases'].index(st.session_state.domain_state.get('phase', domain_cfg['phases'][0]))
    )
    
    st.divider()
    
    # Agent configuration
    st.subheader("LLM Agents")
    agent_models = st.multiselect(
        "Select Active Agents",
        ["Ollama (Local)"],
        default=["Ollama (Local)"]
    )
    st.session_state.agent_pool.active_agents = agent_models
    
    st.divider()
    
    # System info
    st.subheader("System Status")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Events Logged", len(st.session_state.events_log))
    with col2:
        st.metric("Commentary Lines", len(st.session_state.commentary_history))

# Main content area
if selected_video:
    is_live_cam = selected_video == "Live Webcam (0)"
    is_screen_share = selected_video == "Screen Share (Live)"
    is_youtube = selected_video == "YouTube Link (Stream)"
    is_live = is_live_cam or is_screen_share or is_youtube
    
    if is_youtube and not youtube_url:
        st.info("👈 Please enter a YouTube URL in the sidebar to begin")
        st.stop()
    
    video_path = 0 if is_live_cam else os.path.join(os.path.dirname(__file__), '..', 'assets', selected_video)
    
    # Tabs for different views
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📺 Video & Analysis",
        "📊 Event Stream",
        "🤖 Agent Outputs",
        "📈 Domain Dashboard",
        "🔍 Prompt Engineering"
    ])
    
    with tab5:
        st.subheader("LLM Prompt Diagnostics")
        st.write("View the exact micro-prompts being sent to the LLM in real-time.")
        prompt_placeholder = st.empty()

    with tab1:
        st.subheader(f"Video: {selected_video}")
        
        # Player controls on top
        render_match_controls()
        
        col_main, col_data = st.columns([2, 1])
        
        with col_main:
            col_raw, col_analysis = st.columns(2)
            with col_raw:
                st.write("#### Live Stream (Raw)")
                raw_placeholder = st.empty()
            with col_analysis:
                st.write("#### YOLO Analysis (Vision)")
                analysis_placeholder = st.empty()
                
            st.divider()
            st.write("#### Live Commentary")
            comm_placeholder = st.empty()
                
        with col_data:
            st.write("#### DSG JSON Stream")
            json_placeholder = st.empty()
            
        if is_live or os.path.exists(video_path):
            try:
                # Initialize Vision Models
                if 'yolo_detector' not in st.session_state:
                    from perception.detector import Detector
                    st.session_state.yolo_detector = Detector(backend='ultralytics', model_path='yolov8n.pt', device='cuda')
                    
                if 'reltr_generator' not in st.session_state:
                    import sys
                    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
                    from perception.reltr_detector import RelTRSceneGraphGenerator
                    weights = os.path.join(os.path.dirname(__file__), '..', 'reltr_hackathon_checkpoint.pth')
                    if not os.path.exists(weights):
                        weights = None
                    st.session_state.reltr_generator = RelTRSceneGraphGenerator(weights_path=weights)
                    
                sct = None
                cap = None
                
                if is_screen_share:
                    import mss
                    sct = mss.mss()
                    monitor = sct.monitors[1] # Capture primary monitor
                elif is_youtube:
                    import yt_dlp
                    with st.spinner("Extracting stream from YouTube..."):
                        ydl_opts = {'format': 'best[ext=mp4]/best', 'quiet': True}
                        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                            info = ydl.extract_info(youtube_url, download=False)
                            stream_url = info.get('url', youtube_url)
                    cap = cv2.VideoCapture(stream_url)
                else:
                    cap = cv2.VideoCapture(video_path)
                
                if is_live:
                    fps = 30
                    st.session_state.playing = True
                else:
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    
                    # Hidden slider for seeking when paused
                    if not st.session_state.playing:
                        frame_idx = st.slider("Seek Frame", 0, total_frames - 1, st.session_state.current_frame)
                        st.session_state.current_frame = frame_idx
                        
                    cap.set(cv2.CAP_PROP_POS_FRAMES, st.session_state.current_frame)
                
                if st.session_state.playing:
                    while st.session_state.playing:
                        if is_screen_share:
                            sct_img = sct.grab(monitor)
                            frame = np.array(sct_img)
                            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                            ret = True
                        else:
                            if not cap.isOpened():
                                break
                            ret, frame = cap.read()
                            
                        if not ret:
                            st.session_state.playing = False
                            st.rerun()
                            
                        # 1. Vision Analysis & DSG Generation
                        trigger_payload = None
                        frame_nodes = {}
                        
                        if st.session_state.vision_engine == "RelTR Transformer (Experimental)" or "RelTR" in st.session_state.vision_engine:
                            # --- PHASE 3: RELTR SCENE GRAPH GENERATION ---
                            scene_graph = st.session_state.reltr_generator.generate_scene_graph(frame)
                            frame_annotated = st.session_state.reltr_generator.draw_scene_graph(frame.copy(), scene_graph)
                            
                            # Map RelTR graph to UI json format
                            for node in scene_graph['nodes']:
                                frame_nodes[node['label']] = {
                                    "type": node['label'],
                                    "box": node['bbox'],
                                    "center": [(node['bbox'][0]+node['bbox'][2])//2, (node['bbox'][1]+node['bbox'][3])//2],
                                    "velocity_kph": 0
                                }
                            
                            # If RelTR found an edge, fire the payload natively!
                            if scene_graph['edges']:
                                edge = scene_graph['edges'][0]
                                src_node = next(n for n in scene_graph['nodes'] if n['id'] == edge['source'])
                                tgt_node = next(n for n in scene_graph['nodes'] if n['id'] == edge['target'])
                                
                                trigger_payload = {
                                    "event": edge['predicate'].upper(),
                                    "primary_actor": frame_nodes[src_node['label']],
                                    "secondary_actor": frame_nodes[tgt_node['label']]
                                }
                        
                        else:
                            # --- PHASE 1: YOLO + HEURISTIC ENGINE ---
                            detections = st.session_state.yolo_detector.detect(frame, conf=0.5)
                            from perception.detector import draw_detections
                            frame_annotated = draw_detections(frame.copy(), detections) if detections else frame.copy()
                            
                            # Format DSG nodes
                            for i, det in enumerate(detections):
                                cls_name = det['label']
                                
                                # Map standard YOLO COCO classes to our custom Domain labels
                                if cls_name == "sports ball": cls_name = "Ball"
                                elif cls_name in ["baseball bat", "tennis racket"]: cls_name = "Bat"
                                elif cls_name == "person": 
                                    cls_name = "Player"
                                    # --- HEURISTIC ROLE IDENTIFICATION ---
                                    # Check if the person is wearing black (Umpire) or colored jersey (Team)
                                    x1, y1, x2, y2 = map(int, det['box'])
                                    crop = frame[max(0, y1):y2, max(0, x1):x2]
                                    if crop.size > 0:
                                        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
                                        mean_brightness = np.mean(hsv[:, :, 2])
                                        if mean_brightness < 60:
                                            cls_name = "Umpire"
                                
                                x1, y1, x2, y2 = map(int, det['box'])
                                
                                if cls_name not in frame_nodes:
                                    frame_nodes[cls_name] = {
                                        "type": cls_name,
                                        "box": [x1, y1, x2, y2],
                                        "center": [(x1+x2)//2, (y1+y2)//2],
                                        "velocity_kph": 0
                                    }
                            
                            # 2. DSG Engine Processing (Heuristic Math)
                            if st.session_state.dsg_engine and frame_nodes:
                                events = st.session_state.dsg_engine.evaluate_frame(frame_nodes)
                                if events:
                                    trigger_payload = events[0] # Take primary event

                        # --- EVENT-TRIGGERED OCR ---
                        # Only run computationally heavy OCR when an action happens!
                        if trigger_payload:
                            try:
                                if 'ocr_reader' not in st.session_state:
                                    import easyocr
                                    with st.spinner("Loading OCR Model for the first time..."):
                                        st.session_state.ocr_reader = easyocr.Reader(['en'], gpu=True)
                                        
                                h, w = frame.shape[:2]
                                bottom_crop = frame[int(h*0.75):h, :] # Bottom 25% of screen
                                
                                ocr_result = st.session_state.ocr_reader.readtext(bottom_crop, detail=0, paragraph=True)
                                trigger_payload["scoreboard_ocr"] = " ".join(ocr_result)
                            except Exception as e:
                                trigger_payload["scoreboard_ocr"] = f"OCR Error/Missing: {e}"
                            
                            st.session_state.events_log.append({
                                "timestamp": datetime.now().isoformat(),
                                "event": f"Trigger: {trigger_payload.get('event', 'Unknown')}"
                            })
                            
                        # --- FUSIONX TRAFFIC INTELLIGENCE ENGINE ---
                        if st.session_state.selected_domain == "Traffic":
                            if 'traffic_engine' not in st.session_state:
                                from ui.agent_integration import FusionXEngine # We'll use the logic from the previous script
                                st.session_state.traffic_engine = FusionXEngine()
                            
                            # Process frame through the reflex engine
                            frame_annotated = st.session_state.traffic_engine.process_frame(frame, st.session_state.current_frame)
                            
                            # Update Dashboard Components
                            with col_main:
                                st_narration = st.empty()
                                st_narration.markdown(f"""
                                    <div style='background: linear-gradient(90deg, #1e3a8a, #1e40af); padding:15px; border-radius:10px; border-left: 5px solid #3b82f6; margin-bottom:15px;'>
                                        <b style='color:#93c5fd; font-size:0.8rem;'>🎙️ GLOBAL NARRATOR</b><br/>
                                        <span style='font-size:1.1rem; color:white;'>"{st.session_state.traffic_engine.last_narration}"</span>
                                    </div>
                                """, unsafe_allow_html=True)
                            
                            with col_data:
                                st.write("#### 🤖 Agent Swarm Debate")
                                # Physics Agent
                                st.info(f"**Agent 1 (Physics):** {getattr(st.session_state.traffic_engine, 'physics_thought', 'Analyzing kinetic forces...')}")
                                # Auditor Agent
                                st.warning(f"**Agent 2 (Auditor):** {getattr(st.session_state.traffic_engine, 'auditor_thought', 'Evaluating scene context...')}")
                                # Executive Agent
                                st.error(f"**Agent 3 (Executive):** {st.session_state.traffic_engine.last_verdict}")
                                
                                if st.session_state.traffic_engine.instant_alert:
                                    st.markdown("🚨 **INSTANT IMPACT DETECTED**", unsafe_allow_html=True)

                        # --- RENDER ---
                        raw_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_container_width=True)
                        analysis_placeholder.image(cv2.cvtColor(frame_annotated, cv2.COLOR_BGR2RGB), use_container_width=True)
                        json_placeholder.json({"frame": st.session_state.current_frame, "nodes": frame_nodes, "verdict": getattr(st.session_state.traffic_engine, 'last_verdict', 'N/A')})
                            
                        st.session_state.current_frame += 1
                        import time
                        time.sleep(1/fps)
                else:
                    if is_screen_share:
                        sct_img = sct.grab(monitor)
                        frame = np.array(sct_img)
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                        ret = True
                    else:
                        ret, frame = cap.read()
                        
                    if ret:
                        if st.session_state.vision_engine == "RelTR Transformer (Phase 3 Experimental)" or "RelTR" in st.session_state.vision_engine:
                            # Use RelTR for paused frame preview
                            scene_graph = st.session_state.reltr_generator.generate_scene_graph(frame)
                            frame_annotated = st.session_state.reltr_generator.draw_scene_graph(frame.copy(), scene_graph)
                        else:
                            # Use YOLO for paused frame preview
                            detections = st.session_state.yolo_detector.detect(frame, conf=0.5)
                            from perception.detector import draw_detections
                            frame_annotated = draw_detections(frame.copy(), detections) if detections else frame.copy()
                        raw_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_container_width=True)
                        analysis_placeholder.image(cv2.cvtColor(frame_annotated, cv2.COLOR_BGR2RGB), use_container_width=True)
                        json_placeholder.json({"status": "Paused", "frame": st.session_state.current_frame})
                        if st.session_state.commentary_history:
                            html = "<div style='height: 250px; overflow-y: auto; display: flex; flex-direction: column-reverse; padding: 10px; border: 1px solid #333; border-radius: 5px; background: #0e1117;'>"
                            for item in reversed(st.session_state.commentary_history[-15:]):
                                time_str = item['timestamp'][11:19]
                                html += f"<div style='margin-bottom: 8px; padding: 10px; background-color: #1e1e1e; color: white; border-left: 3px solid #ff4b4b;'><strong>[{time_str}] Live:</strong> {item['commentary']}</div>"
                            html += "</div>"
                            comm_placeholder.markdown(html, unsafe_allow_html=True)
                            
                            if hasattr(st.session_state.agent_pool, 'last_system_prompt'):
                                prompt_placeholder.markdown(
                                    f"**System Prompt:**\n```text\n{st.session_state.agent_pool.last_system_prompt}\n```\n\n"
                                    f"**User Prompt:**\n```text\n{st.session_state.agent_pool.last_user_prompt}\n```"
                                )
                
                if cap:
                    cap.release()
            except Exception as e:
                st.error(f"Error reading video: {e}")
        else:
            st.warning(f"Video file not found: {video_path}")
    
    with tab2:
        st.subheader("Event Detection Stream")
        render_event_stream(st.session_state.events_log, st.session_state.domain_state)
    
    with tab3:
        st.subheader("LLM Agent Outputs")
        render_agent_outputs(st.session_state.agent_pool.active_agents, st.session_state.domain_state)
    
    with tab4:
        st.subheader("Domain Dashboard")
        render_match_dashboard(st.session_state.domain_state)
else:
    st.info("👈 Select a video from the sidebar to get started")

# Interactive commentary simulation & Live Sync
st.divider()

col_sync, col_sim = st.columns([1, 1])

with col_sync:
    st.subheader("📡 Live Engine Feed")
    if st.button("🔄 Sync with Engine Layer", use_container_width=True):
        try:
            response = requests.get("http://localhost:8000/history", timeout=2)
            if response.status_code == 200:
                history = response.json().get("history", [])
                st.session_state.commentary_history = history
                st.success(f"Synced {len(history)} events from the Engine!")
                st.rerun()
            else:
                st.error("Engine responded with an error.")
        except requests.exceptions.RequestException:
            st.error("Could not connect to FastAPI Engine. Is it running?")

with col_sim:
    st.subheader("💬 Manual Override Simulation")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        event_description = st.text_area(
            "Describe what happened in the current frame",
            placeholder="e.g., The ball has pitched on good length, outside off stump..."
        )
    
    with col2:
        if st.button("🎙️ Generate Commentary", use_container_width=True):
            if event_description:
                with st.spinner("Generating commentary..."):
                    # Get commentary from agent pool
                    commentary = st.session_state.agent_pool.generate_commentary(
                        event_description,
                        st.session_state.domain_state,
                        st.session_state.selected_domain
                    )
                    
                    if commentary:
                        st.session_state.commentary_history.append({
                            'timestamp': datetime.now().isoformat(),
                            'event': event_description,
                            'commentary': commentary
                        })
                        st.success("✅ Commentary generated!")
                        st.markdown(f"*{commentary}*")
            else:
                st.warning("Please describe an event first")

# Footer
st.divider()
st.markdown(f"""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p><strong>FusionX {st.session_state.selected_domain} Analysis System</strong></p>
    <p>Powered by YOLOv8 + DSG Engine + LLM Agents</p>
    <p><small>Real-time computer vision meets intelligent narration</small></p>
</div>
""", unsafe_allow_html=True)
