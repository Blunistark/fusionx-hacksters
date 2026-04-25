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
import time

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
    page_title="Real-Time Event Engine",
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

    # --- Initialize Domain Engines (Global Setup) ---
    domains_to_warmup = ["Traffic", "Cricket", "Security"]
    for d in domains_to_warmup:
        key = f"{d.lower()}_engine"
        if key not in st.session_state:
            try:
                from agent_integration import FusionXEngine
                st.session_state[key] = FusionXEngine(domain=d)
            except Exception as e:
                pass

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
st.title(f"{domain_cfg['icon']} {st.session_state.selected_domain} Event Engine")

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
    
    # TTS Toggle
    tts_enabled = st.toggle("🎙️ Enable AI Narration", value=True)
    
    # Sync toggle state with engine
    engine_key = f"{st.session_state.selected_domain.lower()}_engine"
    if engine_key in st.session_state:
        st.session_state[engine_key].tts_enabled = tts_enabled
    
    st.divider()

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
    tab1, tab2, tab3 = st.tabs([
        "📺 Video & Analysis",
        "📈 Domain Dashboard",
        "📊 Event Stream"
    ])
    
    with tab1:
        st.subheader(f"Video: {selected_video}")
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
            st_narration = st.empty()
                
        with col_data:
            st.write("#### 📡 Intelligence Feed")
            json_placeholder = st.empty()
            swarm_placeholder = st.empty()
            
        if is_live or os.path.exists(video_path):
            try:
                # Initialize Super Vision Engine
                if 'yolo_detector' not in st.session_state:
                    from perception.detector import SuperDetector
                    from agent_integration import DOMAIN_MODELS
                    domain_models = DOMAIN_MODELS.get(st.session_state.selected_domain, ["yolov8n.pt"]).copy()
                    face_model = "yolov8n-face.pt"
                    if os.path.exists(face_model):
                        domain_models.append(face_model)
                    st.session_state.yolo_detector = SuperDetector(model_paths=domain_models, device='cuda')
                    
                if 'reltr_generator' not in st.session_state:
                    from perception.reltr_detector import RelTRSceneGraphGenerator
                    st.session_state.reltr_generator = RelTRSceneGraphGenerator()
                    
                cap = None
                sct = None
                if is_screen_share:
                    import mss
                    sct = mss.mss()
                    monitor = sct.monitors[1]
                elif is_youtube:
                    import yt_dlp
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
                    cap.set(cv2.CAP_PROP_POS_FRAMES, st.session_state.current_frame)
                
                while True:
                    if not st.session_state.playing:
                        # Paused state handling
                        if is_screen_share:
                            frame = np.array(sct.grab(monitor))
                            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                            ret = True
                        else:
                            ret, frame = cap.read()
                        
                        if ret:
                            raw_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_container_width=True)
                            analysis_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_container_width=True)
                        break
                        
                    # Playing state handling
                    if is_screen_share:
                        frame = np.array(sct.grab(monitor))
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                        ret = True
                    else:
                        ret, frame = cap.read()
                        
                    if not ret:
                        st.session_state.playing = False
                        break
                        
                    # --- VISION & COGNITIVE LOOP ---
                    results = None
                    frame_annotated = frame.copy()
                    frame_nodes = {}
                    
                    if "RelTR" in st.session_state.vision_engine:
                        scene_graph = st.session_state.reltr_generator.generate_scene_graph(frame)
                        frame_annotated = st.session_state.reltr_generator.draw_scene_graph(frame.copy(), scene_graph)
                        current_rels = [f"{n['label']}" for n in scene_graph['nodes']] # Simplification
                    else:
                        results = st.session_state.yolo_detector.detect(frame, conf=0.4)
                        from perception.detector import draw_detections
                        frame_annotated = draw_detections(frame.copy(), results)
                        
                        detections = results.get("detections", [])
                        for det in detections:
                            frame_nodes[det['label']] = {"type": det['label'], "box": det['box']}
                            
                    # --- ENGINE SYNC ---
                    engine_key = f"{st.session_state.selected_domain.lower()}_engine"
                    if engine_key not in st.session_state:
                        from agent_integration import FusionXEngine
                        st.session_state[engine_key] = FusionXEngine(domain=st.session_state.selected_domain)
                    
                    engine = st.session_state[engine_key]
                    if hasattr(engine, 'process_frame_optimized'):
                        engine.process_frame_optimized(frame, st.session_state.current_frame, results=results)
                    
                    # --- RENDER UI ---
                    raw_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_container_width=True)
                    analysis_placeholder.image(cv2.cvtColor(frame_annotated, cv2.COLOR_BGR2RGB), use_container_width=True)
                    json_placeholder.json({"frame": st.session_state.current_frame, "nodes": frame_nodes})
                    
                    st_narration.markdown(f"""
                        <div style='background: linear-gradient(90deg, #1e3a8a, #1e40af); padding:15px; border-radius:10px; border-left: 5px solid #3b82f6;'>
                            <b style='color:#93c5fd;'>🎙️ {st.session_state.selected_domain.upper()} NARRATOR</b><br/>
                            <span style='color:white;'>"{engine.last_narration if engine else 'Initializing...'}"</span>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    with swarm_placeholder.container():
                        st.error(f"**Consensus Verdict:** {engine.last_verdict if engine else 'N/A'}")
                    
                    st.session_state.current_frame += 1
                    time.sleep(0.01)
                    
                if cap: cap.release()
            except Exception as e:
                st.error(f"Engine Error: {e}")
                
    with tab2:
        st.subheader("Domain Dashboard")
        render_match_dashboard(st.session_state.domain_state)
    with tab3:
        st.subheader("Event Detection Stream")
        render_event_stream(st.session_state.events_log, st.session_state.domain_state)

st.divider()
st.markdown(f"<div style='text-align: center; color: #666;'>Powered by FusionX Super-Vision Stack</div>", unsafe_allow_html=True)
