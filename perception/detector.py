import cv2
import numpy as np
import os

class SuperDetector:
    """Multi-Model Super Detector for FusionX.
    
    Supports:
    - Object Detection (YOLOv8/v10/v11)
    - Pose Estimation (Skeletal tracking)
    - Face Detection
    """

    def __init__(self, model_paths=None, device='cpu'):
        try:
            from ultralytics import YOLO
        except ImportError:
            raise RuntimeError("Ultralytics is required for SuperDetector")
            
        self.device = device
        self.models = []
        
        if model_paths is None:
            model_paths = ["yolov8n.pt"]
        elif isinstance(model_paths, str):
            model_paths = [model_paths]
            
        # Always include face detection if not present (optional, but good for "Super" mode)
        # self.models.append(YOLO("yolov8n-face.pt")) 
        
        for path in model_paths:
            print(f"📦 Loading SuperModel: {path}")
            self.models.append(YOLO(path))

    def detect(self, frame, conf=0.5):
        """Run all models on a single frame and aggregate results."""
        all_results = {
            "detections": [],
            "poses": [],
            "faces": []
        }
        
        for model in self.models:
            results = model(frame, conf=conf, verbose=False)
            
            for r in results:
                # Standard Bounding Boxes
                if hasattr(r, 'boxes'):
                    for box in r.boxes:
                        cls_id = int(box.cls[0])
                        label = model.names.get(cls_id, str(cls_id))
                        xyxy = box.xyxy[0].cpu().numpy().tolist()
                        all_results["detections"].append({
                            'box': [int(x) for x in xyxy],
                            'score': float(box.conf[0]),
                            'label': label,
                            'cls_id': cls_id
                        })
                
                # Pose Keypoints
                if hasattr(r, 'keypoints') and r.keypoints is not None:
                    kpts = r.keypoints.data.cpu().numpy() # [N, 17, 3]
                    for person_kpts in kpts:
                        all_results["poses"].append(person_kpts)
        
        return all_results

# Alias for compatibility
YOLODetector = SuperDetector

def draw_detections(frame, results):
    """Advanced drawing for SuperDetector results (Boxes + Poses)."""
    # 1. Draw Bounding Boxes
    for det in results.get("detections", []):
        x1, y1, x2, y2 = det['box']
        label = f"{det['label']} {det['score']:.2f}"
        color = (0, 255, 0) # Default Green
        
        # Color coding by label
        if det['label'] in ['person', 'Player']: color = (255, 128, 0) # Orange
        elif det['label'] in ['bat', 'ball']: color = (0, 0, 255) # Red
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # 2. Draw Poses (Skeletons)
    for kpts in results.get("poses", []):
        # Draw keypoints
        for kp in kpts:
            x, y, conf = kp
            if conf > 0.5:
                cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 255), -1) # Yellow dots
        
        # Draw skeleton lines (Simplified standard YOLO pose skeleton)
        skeleton = [
            (16, 14), (14, 12), (17, 15), (15, 13), (12, 13), (6, 12), (7, 13), (6, 7),
            (6, 8), (7, 9), (8, 10), (9, 11), (2, 3), (1, 2), (1, 3), (2, 4), (3, 5)
        ]
        for start, end in skeleton:
            if start <= len(kpts) and end <= len(kpts):
                p1 = kpts[start-1]
                p2 = kpts[end-1]
                if p1[2] > 0.5 and p2[2] > 0.5:
                    cv2.line(frame, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (0, 255, 0), 2)

    return frame
