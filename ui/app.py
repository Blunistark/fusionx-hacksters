import streamlit as st
import os
import json
import cv2
import numpy as np
from pathlib import Path
import sys
from datetime import datetime

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
        st.rerun()
    
    domain_cfg = DOMAINS[selected_domain]
    st.info(f"{domain_cfg['icon']} {domain_cfg['description']}")
    
    st.divider()
    
    # Video selection
    videos = get_available_videos()
    if videos:
        selected_video = st.selectbox("Select Video", videos)
        st.session_state.selected_video = selected_video
    else:
        st.warning("No videos found in assets folder")
        selected_video = None
    
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
        ["GPT-4o-mini", "Gemini 1.5 Flash", "Claude 3.5"],
        default=["GPT-4o-mini"]
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
    video_path = os.path.join(os.path.dirname(__file__), '..', 'assets', selected_video)
    
    # Tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs([
        "📺 Video & Analysis",
        "📊 Event Stream",
        "🤖 Agent Outputs",
        "📈 Domain Dashboard"
    ])
    
    with tab1:
        st.subheader(f"Video: {selected_video}")
        
        col_video, col_commentary = st.columns([2, 1])
        
        with col_video:
            # Video display section
            st.write("#### Live Video Feed")
            video_placeholder = st.empty()
            
            # Video player controls
            render_match_controls()
            
            # Display video frame or placeholder
            if os.path.exists(video_path):
                try:
                    cap = cv2.VideoCapture(video_path)
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    
                    # Frame slider
                    frame_idx = st.slider(
                        "Frame",
                        0,
                        total_frames - 1,
                        st.session_state.current_frame
                    )
                    st.session_state.current_frame = frame_idx
                    
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                    ret, frame = cap.read()
                    
                    if ret:
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        video_placeholder.image(frame_rgb, use_column_width=True)
                        
                        # Display frame info
                        st.caption(f"Frame {frame_idx + 1} / {total_frames} | FPS: {fps:.1f}")
                    
                    cap.release()
                except Exception as e:
                    st.error(f"Error reading video: {e}")
            else:
                st.warning(f"Video file not found: {video_path}")
        
        with col_commentary:
            st.write("#### Live Commentary")
            
            # Current event display
            if st.session_state.events_log:
                latest_event = st.session_state.events_log[-1]
                st.markdown("**Last Event:**")
                st.info(f"🎯 {latest_event.get('event', 'N/A')}")
            
            # Live commentary display
            render_llm_commentary(st.session_state.commentary_history)
    
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

# Interactive commentary simulation
st.divider()
st.subheader("💬 Generate Live Commentary")

col1, col2 = st.columns([3, 1])

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
