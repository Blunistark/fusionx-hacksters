import streamlit as st
import cv2
import json
import os
import threading
import torch
import time
import requests
from PIL import Image
import yt_dlp
from ultralytics import YOLO
import sys

import requests
import base64

# Add RelTR directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'RelTR')))

# --- NEW: Restored LLMAgentPool for app.py compatibility ---
class LLMAgentPool:
    def __init__(self):
        self.active_agents = ["Ollama (Local)"]
        self.OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
        self.OLLAMA_MODEL = "llama3"

    def generate_commentary(self, event_data, domain_state, domain):
        prompt = f"""
        Domain: {domain}
        State: {domain_state}
        Event: {event_data}
        Task: Provide a professional commentary based on this event. 1 sentence.
        """
        try:
            res = requests.post(self.OLLAMA_URL, json={"model": self.OLLAMA_MODEL, "prompt": prompt, "stream": False}, timeout=10)
            return res.json().get('response', 'Flow continues.')
        except: return "Agent Offline."

# --- DOMAIN INTELLIGENCE CONFIG ---
DOMAIN_PROMPTS = {
    "Cricket": "You are a professional Cricket Commentator. Analyze the player movements and ball trajectory. Speak with excitement and use cricket terminology like 'Good length', 'Square leg', 'Clean bowled'.",
    "Security": "You are a High-Security Warden. Analyze the scene for suspicious behavior, unauthorized entry, or loitering. Be formal and focus on risk assessment.",
    "Traffic": "You are a Traffic Safety Swarm. Analyze for accidents, speeding, and reckless driving. Focus on impact physics and road safety."
}

DOMAIN_MODELS = {
    "Cricket": ["yolov8m.pt", "yolov8m-pose.pt"], # Accuracy + Human Pose
    "Security": ["yolov8x.pt"], # Highest accuracy for faces/objects
    "Traffic": ["yolov8n.pt", "yolov10n.pt"] # High speed for motion
}

class FusionXEngine:
    def __init__(self, domain="Traffic"):
        self.domain = domain
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load the specialized Ensemble for the domain
        self.models = [YOLO(m) for m in DOMAIN_MODELS.get(domain, ["yolov8n.pt"])]
        
        self.last_narration = f"Initializing {domain} Intelligence..."
        self.last_verdict = "Stable"
        self.instant_alert = False
        self.is_thinking = False
        self.OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
        self.OLLAMA_MODEL = "llama3"

    def process_frame(self, frame, frame_count):
        # 1. Run Domain Ensemble
        detections = []
        for model in self.models:
            res = model(frame, verbose=False)[0]
            for box in res.boxes:
                if box.conf[0] > 0.4:
                    label = model.names[int(box.cls[0])]
                    detections.append({"label": label, "bbox": box.xyxy[0].tolist()})
                    cv2.rectangle(frame, (int(box.xyxy[0][0]), int(box.xyxy[0][1])), (int(box.xyxy[0][2]), int(box.xyxy[0][3])), (0, 255, 0), 1)

        # 2. Domain-Specific Reasoning
        self.instant_alert = False
        hazards = []
        
        if self.domain == "Traffic":
            # Impact Physics
            for i, d1 in enumerate(detections):
                for j, d2 in enumerate(detections):
                    if i >= j: continue
                    if abs(d1['bbox'][3] - d2['bbox'][3]) < 40: # Same ground level
                        self.instant_alert = True
                        hazards.append(f"{d1['label']} IMPACT {d2['label']}")

        elif self.domain == "Security":
            if len([d for d in detections if d['label'] == 'person']) > 5:
                self.instant_alert = True
                hazards.append("Unusual Crowd Density Detected")

        # 3. Vision-Aware Brain Call (Every 90 frames for stability)
        if frame_count % 90 == 0 and detections and not self.is_thinking:
            # Convert frame to base64 for Vision analysis
            _, buffer = cv2.imencode('.jpg', frame)
            img_str = base64.b64encode(buffer).decode('utf-8')
            threading.Thread(target=self.ask_vision_ollama, args=(detections, hazards, img_str)).start()

        return frame

    def ask_vision_ollama(self, detections, hazards, img_str):
        self.is_thinking = True
        prompt = f"""
        ROLE: {DOMAIN_PROMPTS[self.domain]}
        JSON DATA: {detections}
        ALERTS: {hazards}
        TASK: Look at the attached image and the JSON data. Confirm the situation and narrate exactly what is happening. Mention visual details like colors or specific actions. 1 short sentence.
        """
        try:
            # Note: We use the 'images' parameter for Vision models in Ollama
            payload = {
                "model": "llava", # Automatically tries llava if available, else falls back
                "prompt": prompt,
                "images": [img_str],
                "stream": False
            }
            res = requests.post(self.OLLAMA_URL, json=payload, timeout=15)
            self.last_narration = res.json().get('response', 'Observing...')
        except:
            # Fallback to text-only if Vision fails
            try:
                res = requests.post(self.OLLAMA_URL, json={"model": self.OLLAMA_MODEL, "prompt": prompt, "stream": False}, timeout=10)
                self.last_narration = res.json().get('response', 'Stable.')
            except: self.last_narration = "Vision Engine Offline."
        self.is_thinking = False

# --- STREAMLIT UI ---
st.set_page_config(page_title="FusionX Command Center", layout="wide")
st.markdown("<style>.stApp { background-color: #0e1117; color: white; }</style>", unsafe_allow_html=True)

if 'engine' not in st.session_state:
    st.session_state.engine = FusionXEngine()

def get_youtube_url(url):
    ydl_opts = {'format': 'best', 'quiet': True}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        return ydl.extract_info(url, download=False)['url']

st.title("🛡️ FusionX Global Command Center")
source_type = st.sidebar.selectbox("Video Source", ["YouTube Link", "Local File", "Live Camera"])
source_url = st.sidebar.text_input("Enter URL/Path", "../assets/accident-1.mp4")

run_engine = st.sidebar.button("🚀 START FUSION ENGINE")
stop_engine = st.sidebar.button("🛑 STOP ENGINE")

if run_engine:
    final_url = source_url
    if source_type == "YouTube Link" and source_url.startswith("http"):
        try: final_url = get_youtube_url(source_url)
        except Exception as e: st.error(f"YouTube Error: {e}"); st.stop()

    cap = cv2.VideoCapture(final_url)
    st_frame = st.empty()
    st_narration = st.empty()
    
    col1, col2 = st.columns([2, 1])
    with col2:
        st.subheader("📸 Impact Evidence Gallery")
        evidence_placeholder = st.empty()

    frame_count = 0
    while cap.isOpened() and not stop_engine:
        ret, frame = cap.read()
        if not ret: break
        
        # Process every 2nd frame to keep UI smooth
        if frame_count % 2 == 0:
            processed_frame = st.session_state.engine.process_frame(frame, frame_count)
            st_frame.image(processed_frame, channels="BGR", use_column_width=True)
            
            st_narration.markdown(f"""
                <div style='background:#1e3a8a; padding:15px; border-radius:10px; border-left: 5px solid #3b82f6;'>
                    <b style='color:#93c5fd;'>🎙️ AI NARRATOR:</b> {st.session_state.engine.last_narration}
                </div>
            """, unsafe_allow_html=True)
            
            if st.session_state.engine.evidence_captured:
                latest = st.session_state.engine.evidence_captured[-1]
                evidence_placeholder.image(latest['path'], caption=f"ALERT: {latest['reason']} @ {latest['time']}")

        frame_count += 1
        time.sleep(0.01) # Small delay to let Streamlit breathe

    cap.release()
    st.success("Engine Stopped Safely.")
