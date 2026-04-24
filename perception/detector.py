import cv2
import numpy as np

class Detector:
    """Model-agnostic detector adapter.

    Backends supported:
      - "ultralytics" (YOLOv8/9 via `ultralytics` package)
      - "yolov5" (torch.hub ultralytics/yolov5)

    Usage:
      det = Detector(backend='ultralytics', model_path='yolov8n.pt', device='cpu')
      detections = det.detect(frame, conf=0.5)
    Each detection is a dict: { 'box': [x1,y1,x2,y2], 'score': float, 'label': str, 'cls_id': int }
    """

    def __init__(self, backend='ultralytics', model_path=None, device='cpu'):
        self.backend = backend
        self.model_path = model_path
        self.device = device
        self.model = None
        self.names = {}
        self._load_model()

    def _load_model(self):
        if self.backend == 'ultralytics':
            try:
                from ultralytics import YOLO
            except Exception as e:
                raise RuntimeError('ultralytics backend selected but ultralytics is not installed') from e

            # If no model_path provided, default to yolov8n
            model_path = self.model_path or 'yolov8n.pt'
            self.model = YOLO(model_path)
            # names mapping
            try:
                self.names = self.model.names
            except Exception:
                self.names = {i: str(i) for i in range(1000)}

        elif self.backend == 'yolov5':
            try:
                import torch
            except Exception as e:
                raise RuntimeError('yolov5 backend selected but torch is not installed') from e

            # Load from torch hub
            model_name = self.model_path or 'yolov5s'
            self.model = torch.hub.load('ultralytics/yolov5', model_name, pretrained=True)
            self.model.to(self.device)
            # yolov5 uses .names
            try:
                self.names = self.model.names
            except Exception:
                self.names = {i: str(i) for i in range(1000)}
        else:
            raise ValueError(f'Unsupported backend: {self.backend}')

    def detect(self, frame, conf=0.5):
        """Run detection on a single BGR frame. Returns list of detection dicts."""
        if self.backend == 'ultralytics':
            results = self.model(frame, conf=conf, verbose=False)
            detections = []
            for result in results:
                for box in result.boxes:
                    try:
                        xyxy = box.xyxy[0].cpu().numpy().tolist()
                    except Exception:
                        # Fallback if already numpy
                        xyxy = box.xyxy[0].numpy().tolist()

                    score = float(box.conf[0]) if hasattr(box, 'conf') else float(box.conf)
                    cls_id = int(box.cls[0])
                    label = self.names.get(cls_id, str(cls_id))
                    detections.append({
                        'box': [int(x) for x in xyxy],
                        'score': float(score),
                        'label': label,
                        'cls_id': cls_id
                    })

            return detections

        elif self.backend == 'yolov5':
            # yolov5 returns a results object with .xyxy[0] as numpy array
            results = self.model(frame)
            detections = []
            xyxy = results.xyxy[0].cpu().numpy() if hasattr(results.xyxy[0], 'cpu') else results.xyxy[0].numpy()
            for det in xyxy:
                x1, y1, x2, y2, score, cls_id = det.tolist()
                cls_id = int(cls_id)
                label = self.names.get(cls_id, str(cls_id))
                detections.append({
                    'box': [int(x1), int(y1), int(x2), int(y2)],
                    'score': float(score),
                    'label': label,
                    'cls_id': cls_id
                })

            return detections

        else:
            return []


def draw_detections(frame, detections):
    """Utility to draw boxes and labels on a frame (in-place)."""
    for det in detections:
        x1, y1, x2, y2 = det['box']
        label = f"{det['label']} {det['score']:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return frame
