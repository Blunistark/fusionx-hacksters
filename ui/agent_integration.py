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
    "Cricket": "You are an Expert Cricket Commentator on LIVE TV. Provide high-energy, technical commentary. NEVER use phrases like 'In the image', 'The picture shows', 'I can see', or 'Based on the image'. Speak directly about the action. Use terms like 'striker', 'good length', 'crease'.",
    "Security": "You are a Senior Security Analyst. Provide formal, precise threat assessments. Focus on perimeter integrity and suspicious patterns. NEVER mention 'the image', 'the photo', or 'the camera view'. Maintain a professional, alert tone.",
    "Traffic": "You are a Technical Incident Reporter. Analyze traffic density, accidents, and safety hazards. NEVER provide suggestions, advice, or city planning recommendations. Focus ONLY on live action: congestion peaks, vehicle maneuvers, and immediate risks. Speak as if you are observing the live scene directly.",
    "Swarm": "You are the  Consensus Engine. Summarize the state into a concise status code and a 5-word summary. NEVER use descriptive filler. Format: [STATUS] - [BRIEFING]"
}

DOMAIN_MODELS = {
    "Cricket": ["yolov8m.pt", "yolov8m-pose.pt", "yolov8s.pt"], 
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
        
        self.last_narration = "Awaiting Vision Data..."
        self.last_verdict = "Stable"
        self.instant_alert = False
        self.is_thinking = False
        self.SARVAM_API_KEY = os.getenv("SARVAM_API_KEY", "sk_ikb2i84u_feROSQZbrdCM3BBd3uFu5kB0")
        self.SARVAM_URL = "https://api.sarvam.ai/text-to-speech"
        self.tts_enabled = True
        self.OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
        self.OLLAMA_MODEL = "llama3"

    def process_frame_optimized(self, frame, frame_count, results=None, rels=None):
        """Optimized version that receives pre-calculated detections from the SuperDetector"""
        if results is None:
            return self.process_frame(frame, frame_count, rels)
            
        detections = results.get("detections", [])
        self.instant_alert = False
        hazards = []
        
        # Hazard Detection Logic (Spatial Heuristics)
        if self.domain == "Traffic":
            for i, d1 in enumerate(detections):
                for j, d2 in enumerate(detections):
                    if i >= j: continue
                    if abs(d1['box'][3] - d2['box'][3]) < 40: # Y-axis proximity check
                        self.instant_alert = True
                        hazards.append(f"{d1['label']} IMPACT {d2['label']}")
        elif self.domain == "Security":
            if len([d for d in detections if d['label'] == 'person']) > 5:
                self.instant_alert = True
                hazards.append("Crowd Density Alert")

        # Human-like frequency: Every 900 frames (~30s)
        if frame_count % 900 == 0 and detections and not self.is_thinking:
            _, buffer = cv2.imencode('.jpg', frame)
            img_str = base64.b64encode(buffer).decode('utf-8')
            threading.Thread(target=self.ask_vision_ollama, args=(detections, hazards, img_str, rels)).start()

        return frame

    def process_frame(self, frame, frame_count, rels=None):
        detections = []
        for model in self.models:
            res = model(frame, verbose=False)[0]
            for box in res.boxes:
                if box.conf[0] > 0.4:
                    label = model.names[int(box.cls[0])]
                    detections.append({"label": label, "bbox": box.xyxy[0].tolist()})

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

        # Human-like frequency: Every 900 frames (~30s)
        if frame_count % 900 == 0 and detections and not self.is_thinking:
            _, buffer = cv2.imencode('.jpg', frame)
            img_str = base64.b64encode(buffer).decode('utf-8')
            threading.Thread(target=self.ask_vision_ollama, args=(detections, hazards, img_str, rels)).start()

        return frame

    def ask_vision_ollama(self, detections, hazards, img_str, rels=None):
        self.is_thinking = True
        prompt = f"ROLE: {DOMAIN_PROMPTS[self.domain]}\nDATA: {detections}\nRELATIONSHIPS: {rels}\nALERTS: {hazards}\nTASK: You are on live TV. Provide a 1-sentence expert commentary of the CURRENT ACTION. Do not mention that you are looking at an image."
        try:
            payload = {"model": "llava", "prompt": prompt, "images": [img_str], "stream": False}
            res = requests.post(self.OLLAMA_URL, json=payload, timeout=15)
            self.last_narration = res.json().get('response', 'Observing...')
            
            # --- SWARM CONSENSUS UPDATE ---
            swarm_prompt = f"DATA: {detections}\nALERTS: {hazards}\nTASK: {DOMAIN_PROMPTS['Swarm']}\nRespond in Format: [STATUS] - [SUMMARY]"
            try:
                swarm_payload = {"model": self.OLLAMA_MODEL, "prompt": swarm_prompt, "stream": False}
                swarm_res = requests.post(self.OLLAMA_URL, json=swarm_payload, timeout=5)
                self.last_verdict = swarm_res.json().get('response', 'Stable')
            except:
                self.last_verdict = "STABLE - System Nominal" if not hazards else f"CRITICAL - {hazards[0]}"
                
            self.speak_sync(self.last_narration)
        except:
            try:
                res = requests.post(self.OLLAMA_URL, json={"model": self.OLLAMA_MODEL, "prompt": prompt, "stream": False}, timeout=10)
                self.last_narration = res.json().get('response', 'Stable.')
                self.speak_sync(self.last_narration)
            except: self.last_narration = "Brain Offline."
        self.is_thinking = False

    def speak_sync(self, text):
        """Sarvam AI TTS Sync for UI"""
        if not self.SARVAM_API_KEY or not self.tts_enabled:
            print(f"🔇 [UI TTS DISABLED]: {text}")
            return
            
        print(f"🔊 [SARVAM AI SYNCING]: {text}")
        threading.Thread(target=self.call_sarvam_tts, args=(text,)).start()

    def call_sarvam_tts(self, text):
        payload = {
            "inputs": [text],
            "target_language_code": "en-IN",
            "speaker": "meera",
            "pitch": 0,
            "pace": 1.1,
            "loudness": 1.5,
            "speech_sample_rate": 22050,
            "enable_preprocessing": True,
            "model": "bulbul:v1"
        }
        headers = {"api-subscription-key": self.SARVAM_API_KEY}
        
        try:
            res = requests.post(self.SARVAM_URL, json=payload, headers=headers, timeout=10)
            if res.status_code == 200:
                audio_base64 = res.json().get('audios', [None])[0]
                if audio_base64:
                    import base64
                    self.current_audio = base64.b64decode(audio_base64)
            else:
                print(f"⚠️ Sarvam AI TTS Error: {res.status_code}")
        except: pass

    def get_audio(self):
        """Retrieve and clear current audio buffer"""
        audio = getattr(self, 'current_audio', None)
        if audio:
            self.current_audio = None
        return audio
