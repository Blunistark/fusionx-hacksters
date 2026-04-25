import cv2
import json
import os
import threading
import torch
import time
import requests
import base64
import sys
from ultralytics import YOLO

# Add RelTR directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'RelTR')))

# --- DOMAIN INTELLIGENCE CONFIG ---
DOMAIN_PROMPTS = {
    "Cricket": "You are a professional Cricket Commentator. Analyze the player movements and ball trajectory. Speak with excitement and use cricket terminology like 'Good length', 'Square leg', 'Clean bowled'.",
    "Security": "You are a High-Security Warden. Analyze the scene for suspicious behavior, unauthorized entry, or loitering. Be formal and focus on risk assessment.",
    "Traffic": "You are a Traffic Safety Swarm. Analyze for accidents, speeding, and reckless driving. Focus on impact physics and road safety."
}

DOMAIN_MODELS = {
    "Cricket": ["yolov8m.pt", "yolov8m-pose.pt"], 
    "Security": ["yolov8x.pt"], 
    "Traffic": ["yolov8n.pt", "yolov10n.pt"]
}

# --- LLM AGENT POOL (Legacy Compatibility) ---
class LLMAgentPool:
    def __init__(self):
        self.active_agents = ["Ollama (Local)"]
        self.OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
        self.OLLAMA_MODEL = "llama3"

    def generate_commentary(self, event_data, domain_state, domain):
        prompt = f"Domain: {domain}\nEvent: {event_data}\nTask: 1-sentence professional commentary."
        try:
            res = requests.post(self.OLLAMA_URL, json={"model": self.OLLAMA_MODEL, "prompt": prompt, "stream": False}, timeout=10)
            return res.json().get('response', 'Analysis complete.')
        except: return "Agent Offline."

# --- MAIN INTELLIGENCE ENGINE ---
class FusionXEngine:
    def __init__(self, domain="Traffic"):
        self.domain = domain
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.models = [YOLO(m) for m in DOMAIN_MODELS.get(domain, ["yolov8n.pt"])]
        
        self.last_narration = f"Initializing {domain} Intelligence..."
        self.last_verdict = "Stable"
        self.instant_alert = False
        self.is_thinking = False
        self.OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
        self.OLLAMA_MODEL = "llama3"

    def process_frame(self, frame, frame_count):
        detections = []
        for model in self.models:
            res = model(frame, verbose=False)[0]
            for box in res.boxes:
                if box.conf[0] > 0.4:
                    label = model.names[int(box.cls[0])]
                    detections.append({"label": label, "bbox": box.xyxy[0].tolist()})
                    cv2.rectangle(frame, (int(box.xyxy[0][0]), int(box.xyxy[0][1])), (int(box.xyxy[0][2]), int(box.xyxy[0][3])), (0, 255, 0), 1)

        self.instant_alert = False
        hazards = []
        
        if self.domain == "Traffic":
            for i, d1 in enumerate(detections):
                for j, d2 in enumerate(detections):
                    if i >= j: continue
                    if abs(d1['bbox'][3] - d2['bbox'][3]) < 40:
                        self.instant_alert = True
                        hazards.append(f"{d1['label']} IMPACT {d2['label']}")

        elif self.domain == "Security":
            if len([d for d in detections if d['label'] == 'person']) > 5:
                self.instant_alert = True
                hazards.append("Crowd Density Alert")

        if frame_count % 90 == 0 and detections and not self.is_thinking:
            _, buffer = cv2.imencode('.jpg', frame)
            img_str = base64.b64encode(buffer).decode('utf-8')
            threading.Thread(target=self.ask_vision_ollama, args=(detections, hazards, img_str)).start()

        return frame

    def ask_vision_ollama(self, detections, hazards, img_str):
        self.is_thinking = True
        prompt = f"ROLE: {DOMAIN_PROMPTS[self.domain]}\nDATA: {detections}\nALERTS: {hazards}\nTASK: Describe the scene visually. 1 sentence."
        try:
            payload = {"model": "llava", "prompt": prompt, "images": [img_str], "stream": False}
            res = requests.post(self.OLLAMA_URL, json=payload, timeout=15)
            self.last_narration = res.json().get('response', 'Observing...')
        except:
            try:
                res = requests.post(self.OLLAMA_URL, json={"model": self.OLLAMA_MODEL, "prompt": prompt, "stream": False}, timeout=10)
                self.last_narration = res.json().get('response', 'Stable.')
            except: self.last_narration = "Brain Offline."
        self.is_thinking = False
