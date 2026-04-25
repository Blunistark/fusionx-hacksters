import cv2
import json
import requests
import os
import threading
import torch
import time
import torchvision.transforms as T
from PIL import Image
from collections import deque
from ultralytics import YOLO

# Add current dir to path to import official modules
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from models import build_model

class FusionXUltraMaster:
    def __init__(self, domain="Cricket"):
        print(f"--- INITIALIZING ULTRA MASTER ({domain} Mode) ---")
        self.domain = domain
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.yolo_a = YOLO('yolov8n.pt')
        self.yolo_b = YOLO('yolov10n.pt')
        self.frame_count = 0
        
        # Memory & Logic
        self.short_term_history = deque(maxlen=150)
        self.last_swarm_verdict = "System Armed"
        self.last_narration = "Analyzing Session..."
        self.OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
        self.OLLAMA_MODEL = "llama3"
        self.is_thinking = False
        self.is_narrating = False
        self.instant_alert = False
        self.current_rels = []
        self.long_term_memory = deque(maxlen=120) # 1 hour context (120 * 30s)
        self.is_summarizing = False
        self.SARVAM_API_KEY = os.getenv("SARVAM_API_KEY", "sk_ikb2i84u_feROSQZbrdCM3BBd3uFu5kB0")
        self.SARVAM_URL = "https://api.sarvam.ai/text-to-speech"
        
        # Domain-Specific Narrator Personalities
        self.DOMAIN_CONFIGS = {
            "Cricket": {
                "role": "Expert Cricket Commentator on LIVE TV",
                "task": "Provide a high-energy, technical 1-sentence summary of the LIVE action. NEVER use phrases like 'In the image' or 'The player is seen'. Speak as if you are on air right now.",
            },
            "Traffic": {
                "role": "Traffic Safety Swarm",
                "task": "Summarize the last 5s into a 1-sentence technical traffic flow and safety report.",
            },
            "Security": {
                "role": "High-Security Warden",
                "task": "Provide a 1-sentence formal risk assessment focused on unauthorized movement or suspicious patterns.",
            }
        }
        
        # Logging & Evidence (Linked to UI)
        self.log_file = "FusionX_Intelligence.json"
        self.evidence_dir = "accident_evidence"
        if not os.path.exists(self.evidence_dir): os.makedirs(self.evidence_dir)
        self.intelligence_logs = []

        # RelTR Radar Setup
        class Args:
            backbone = 'resnet50'; dilation = False; position_embedding = 'sine'; 
            enc_layers = 6; dec_layers = 6; dim_feedforward = 2048; hidden_dim = 256; 
            dropout = 0.1; nheads = 8; num_entities = 100; num_triplets = 200; 
            aux_loss = False; set_cost_class = 1; set_cost_bbox = 5; set_cost_giou = 2; 
            set_cost_obj_class = 1; set_cost_rel_class = 1; bbox_loss_coef = 5; 
            giou_loss_coef = 2; rel_loss_coef = 1; eos_coef = 0.1; entity_loss_coef = 1;
            dataset = 'vg'; device = 'cuda' if torch.cuda.is_available() else 'cpu';
            lr_backbone = 0; masks = False; return_interm_layers = False; 
            frozen_weights = None; pre_norm = False; set_iou_threshold = 0.5;
        self.reltr, _, _ = build_model(Args())
        self.reltr.to(self.device)
        self.has_reltr = False
        if os.path.exists('ckp/checkpoint0149.pth'):
            checkpoint = torch.load('ckp/checkpoint0149.pth', map_location=self.device, weights_only=False)
            self.reltr.load_state_dict(checkpoint['model'])
            self.reltr.eval(); self.has_reltr = True
            
        self.transform = T.Compose([T.Resize(800), T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.REL_CLASSES = ["bg", "above", "across", "against", "along", "and", "at", "attached to", "behind", "belonging to", "between", "carrying", "covered in", "covering", "eating", "flying in", "for", "from", "gazing at", "hanging from", "has", "holding", "in", "in front of", "incorporating", "looking at", "lying on", "near", "of", "on", "on back of", "over", "painted on", "parked on", "part of", "playing", "riding", "says", "sitting on", "standing on", "street", "through", "to", "toward", "under", "using", "walking in", "walking on", "watching", "wearing", "with"]

    def save_state(self, frame, detections, hazards):
        """Saves everything for the Streamlit UI to read."""
        timestamp = int(time.time())
        img_name = f"evidence_{self.frame_count}.jpg"
        img_path = os.path.join(self.evidence_dir, img_name)
        if self.instant_alert: cv2.imwrite(img_path, frame)
        
        log_entry = {
            "timestamp": time.ctime(),
            "frame": self.frame_count,
            "last_narration": self.last_narration,
            "swarm_verdict": self.last_swarm_verdict,
            "objs": [d['label'] for d in detections],
            "hazards": hazards,
            "is_accident": self.instant_alert,
            "evidence_img": img_path if self.instant_alert else None
        }
        self.intelligence_logs.append(log_entry)
        with open(self.log_file, "w") as f: json.dump(self.intelligence_logs[-50:], f, indent=4)

    def ask_swarm_async(self, frame_copy, detections, rels, hazards):
        self.is_thinking = True
        config = self.DOMAIN_CONFIGS.get(self.domain, self.DOMAIN_CONFIGS["Cricket"])
        prompt = f"ROLE: {config['role']}\nSWARM TASK: Analyze Objs={detections}, Rels={rels}, Hazards={hazards}. {config['task']} 1 short verdict."
        try:
            res = requests.post(self.OLLAMA_URL, json={"model": self.OLLAMA_MODEL, "prompt": prompt, "stream": False}, timeout=5)
            self.last_swarm_verdict = res.json().get('response', 'System stable.')
            
            # Add to long term memory if significant
            if self.instant_alert:
                self.long_term_memory.append(f"ALERT: {self.last_swarm_verdict}")
                
            self.save_state(frame_copy, detections, hazards)
        except: pass
        self.is_thinking = False

    def ask_narrator_async(self):
        self.is_narrating = True
        config = self.DOMAIN_CONFIGS.get(self.domain, self.DOMAIN_CONFIGS["Cricket"])
        
        # Build Cognitive History Context
        history_summary = list(self.long_term_memory)[-10:] # Last 10 significant summaries
        context = {
            "recent_5s_events": self.intelligence_logs[-5:],
            "detected_relationships": self.current_rels,
            "hourly_narrative_context": history_summary
        }
        
        prompt = f"ROLE: {config['role']}\nTASK: {config['task']} (Make it flow naturally with previous context)\nCONTEXT: {context}"
        try:
            res = requests.post(self.OLLAMA_URL, json={"model": self.OLLAMA_MODEL, "prompt": prompt, "stream": False}, timeout=10)
            self.last_narration = res.json().get('response', 'Continuing analysis.')
            self.long_term_memory.append(self.last_narration)
            self.speak_commentary(self.last_narration)
        except: pass
        self.is_narrating = False

    def speak_commentary(self, text):
        """Sarvam AI TTS Integration"""
        if not self.SARVAM_API_KEY:
            print(f"🔇 [TTS DISABLED - NO API KEY]: {text}")
            return
            
        print(f"🎙️ [SARVAM AI SYNCING]: {text}")
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
                    import winsound
                    audio_data = base64.b64decode(audio_base64)
                    temp_audio = "commentary_temp.wav"
                    with open(temp_audio, "wb") as f:
                        f.write(audio_data)
                    winsound.PlaySound(temp_audio, winsound.SND_FILENAME | winsound.SND_ASYNC)
            else:
                print(f"⚠️ Sarvam AI TTS Error: {res.status_code}")
        except Exception as e:
            print(f"❌ Sarvam AI TTS Failed: {str(e)}")

    def run(self, video_path):
        cap = cv2.VideoCapture(video_path)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            res_a = self.yolo_a(frame, verbose=False)[0]
            res_b = self.yolo_b(frame, verbose=False)[0]
            detections = []
            for r in [res_a, res_b]:
                for box in r.boxes:
                    if box.conf[0] > 0.45:
                        detections.append({"label": r.names[int(box.cls[0])], "bbox": box.xyxy[0].tolist()})

            self.instant_alert = False
            hazards = []
            if self.frame_count % 30 == 0:
                current_rels = []
                if self.has_reltr:
                    img = self.transform(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))).unsqueeze(0).to(self.device)
                    with torch.no_grad():
                        out = self.reltr(img)
                        conf, labels = out['rel_logits'].softmax(-1)[0, :, :-1].max(-1)
                        self.current_rels = []
                        for i in torch.where(conf > 0.35)[0]: self.current_rels.append(self.REL_CLASSES[labels[i].item()])
                
                current_rels = self.current_rels
                
                for i, d1 in enumerate(detections):
                    for j, d2 in enumerate(detections):
                        if i >= j: continue
                        b1, b2 = d1['bbox'], d2['bbox']
                        if not (b1[2] < b2[0] or b1[0] > b2[2] or b1[3] < b2[1] or b1[1] > b2[3]):
                            if abs(b1[3] - b2[3]) < 45:
                                self.instant_alert = True
                                hazards.append(f"{d1['label']} IMPACT {d2['label']}")

                if not self.is_thinking:
                    threading.Thread(target=self.ask_swarm_async, args=(frame.copy(), detections, current_rels, hazards)).start()

            if self.frame_count % 900 == 0 and not self.is_narrating and self.intelligence_logs:
                threading.Thread(target=self.ask_narrator_async).start()

            # Dashboard Visuals
            cv2.rectangle(frame, (0, 0), (frame.shape[1], 110), (10, 10, 10), -1)
            cv2.putText(frame, f"NARRATOR: {self.last_narration[:90]}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            color = (0, 0, 255) if self.instant_alert else (0, 255, 0)
            cv2.putText(frame, f"SWARM: {self.last_swarm_verdict[:90]}", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            if self.instant_alert: cv2.rectangle(frame, (0,0), (frame.shape[1], frame.shape[0]), (0,0,255), 10)
            cv2.imshow("FusionX ULTRA MASTER Dashboard", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
            self.frame_count += 1
            
        cap.release(); cv2.destroyAllWindows()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", type=str, default="Cricket", choices=["Cricket", "Traffic", "Security"])
    parser.add_argument("--video", type=str, default="../assets/cricket_demo.mp4")
    args = parser.parse_args()
    
    runner = FusionXUltraMaster(domain=args.domain)
    runner.run(args.video)
