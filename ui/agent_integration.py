import streamlit as st
import cv2
import json
import os
import threading
import torch
import time
import torchvision.transforms as T
from PIL import Image
import yt_dlp
from ultralytics import YOLO
import sys

# Add local path for RelTR models
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'perception')))
from models import build_model

# --- AI CORE ENGINE ---
class FusionXEngine:
    def __init__(self, model_name='yolov8n.pt'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = YOLO(model_name)
        self.last_narration = "Awaiting Stream..."
        self.last_verdict = "System Ready"
        self.instant_alert = False
        self.evidence_captured = []
        self.OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
        self.OLLAMA_MODEL = "llama3"

    def process_frame(self, frame, frame_count):
        # 1. YOLO Detection
        results = self.model(frame, verbose=False)[0]
        detections = []
        for box in results.boxes:
            if box.conf[0] > 0.45:
                label = self.model.names[int(box.cls[0])]
                bbox = box.xyxy[0].tolist()
                detections.append({"label": label, "bbox": bbox})
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 1)

        # 2. Physics Check
        self.instant_alert = False
        for i, d1 in enumerate(detections):
            for j, d2 in enumerate(detections):
                if i >= j: continue
                b1, b2 = d1['bbox'], d2['bbox']
                if not (b1[2] < b2[0] or b1[0] > b2[2] or b1[3] < b2[1] or b1[1] > b2[3]):
                    if abs(b1[3] - b2[3]) < 40:
                        self.instant_alert = True
                        if frame_count % 30 == 0:
                            self.capture_evidence(frame, d1['label'], d2['label'])
        
        # 3. Ollama Reasoning every 60 frames
        if frame_count % 60 == 0 and detections:
            threading.Thread(target=self.ask_ollama, args=(detections,)).start()

        return frame

    def capture_evidence(self, frame, label1, label2):
        ts = int(time.time())
        path = f"accident_evidence/ui_capture_{ts}.jpg"
        if not os.path.exists("accident_evidence"): os.makedirs("accident_evidence")
        cv2.imwrite(path, frame)
        self.evidence_captured.append({"path": path, "reason": f"{label1} vs {label2}"})

    def ask_ollama(self, detections):
        prompt = f"Traffic Data: {detections}. 1 sentence status."
        try:
            res = requests.post(self.OLLAMA_URL, json={"model": self.OLLAMA_MODEL, "prompt": prompt, "stream": False}, timeout=5)
            self.last_narration = res.json().get('response', 'Secure.')
        except: self.last_narration = "Brain Offline."

# --- STREAMLIT UI ---
st.set_page_config(page_title="FusionX Command Center", layout="wide")
st.markdown("<style>.stApp { background-color: #0e1117; color: white; }</style>", unsafe_allow_html=True)

if 'engine' not in st.session_state:
    st.session_state.engine = FusionXEngine()

def get_youtube_url(url):
    ydl_opts = {'format': 'best', 'quiet': True}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        return info['url']

st.title("🛡️ FusionX Global Command Center")
source_type = st.sidebar.selectbox("Video Source", ["YouTube Link", "Local File", "Live Camera"])
source_url = st.sidebar.text_input("Enter URL/Path", "../assets/accident-1.mp4")

if st.sidebar.button("🚀 START FUSION ENGINE"):
    st.info("Initializing GPU Engine...")
    
    if source_type == "YouTube Link":
        final_url = get_youtube_url(source_url)
    else:
        final_url = source_url

    cap = cv2.VideoCapture(final_url)
    st_frame = st.empty()
    st_narration = st.empty()
    
    col1, col2 = st.columns([2, 1])
    with col2:
        st.subheader("📸 Impact Evidence")
        evidence_placeholder = st.empty()

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        # AI Processing
        processed_frame = st.session_state.engine.process_frame(frame, frame_count)
        
        # UI Updates
        st_frame.image(processed_frame, channels="BGR", use_column_width=True)
        st_narration.markdown(f"""
            <div style='background:#1e3a8a; padding:15px; border-radius:10px;'>
                <b>🎙️ AI NARRATOR:</b> {st.session_state.engine.last_narration}
            </div>
        """, unsafe_allow_html=True)
        
        # Show evidence if captured
        if st.session_state.engine.evidence_captured:
            with col2:
                latest = st.session_state.engine.evidence_captured[-1]
                evidence_placeholder.image(latest['path'], caption=f"ALERT: {latest['reason']}")

        frame_count += 1
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
