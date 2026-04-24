import cv2
import json
import time
import requests
from ultralytics import YOLO

class PerceptionLayer:
    def __init__(self, video_source, model_path="yolov8n.pt", stream_delay=0.03, engine_url="http://localhost:8000/ingest"):
        """
        Initializes the YOLO model and the video capture.
        For a hackathon MVP, we use the nano model (yolov8n.pt) for maximum FPS.
        """
        print(f"Loading YOLO model from {model_path}...")
        self.model = YOLO(model_path)
        self.video_source = video_source
        self.cap = cv2.VideoCapture(self.video_source)
        self.stream_delay = stream_delay # Simulate real-time if reading from file
        self.engine_url = engine_url

    def extract_physics(self, box, cls_name):
        """
        Extracts raw geometry from the YOLO bounding box.
        Returns a dictionary formatted for our DSG engine.
        """
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        # Calculate center point for velocity tracking (simplified for MVP)
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        
        return {
            "type": cls_name,
            "box": [x1, y1, x2, y2],
            "center": [cx, cy],
            # For the MVP, we will mock velocity. In Phase 2, calculate this using previous frames.
            "velocity_kph": 0 
        }

    def send_to_engine(self, frame_data):
        """
        Pushes the extracted coordinate array to the FastAPI Engine via HTTP POST.
        This connects Layer 1 (Perception) to Layer 2 (The DSG FSM).
        """
        try:
            response = requests.post(self.engine_url, json=frame_data, timeout=0.1)
            # print(f"[DATA STREAM] Sent to Engine | Status: {response.status_code}")
        except requests.exceptions.RequestException as e:
            print(f"[DATA STREAM ERROR] Could not connect to Engine at {self.engine_url}")

    def run(self):
        """
        The main ingestion loop. Runs at the Speed of Sight (60 FPS ideally).
        """
        if not self.cap.isOpened():
            print(f"Error: Could not open video source {self.video_source}")
            return

        frame_count = 0
        print("Starting live ingestion pipeline...")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break # End of video or stream drop
                
            frame_count += 1
            
            # Run YOLO inference
            # conf=0.5 ensures we only get confident detections, reducing noise for the FSM
            results = self.model(frame, verbose=False, conf=0.5)
            
            frame_nodes = {}
            
            # Parse YOLO results into our DSG node format
            for result in results:
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    cls_name = self.model.names[cls_id]
                    
                    # For cricket, you'd filter for 'sports ball', 'person' (bat/batsman)
                    # For this generic script, we parse whatever it confidently detects
                    node_data = self.extract_physics(box, cls_name)
                    
                    # Assign a unique ID per class in this frame (simplified)
                    node_id = f"{cls_name}_{len(frame_nodes)}"
                    frame_nodes[node_id] = node_data

            # Push the parsed coordinates to the middleware immediately
            if frame_nodes:
                self.send_to_engine({"frame": frame_count, "nodes": frame_nodes})
            
            # Optional: Display the frame for debugging during the hackathon
            cv2.imshow("Layer 1: Perception (YOLO Vision)", results[0].plot())
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
            time.sleep(self.stream_delay)

        self.cap.release()
        cv2.destroyAllWindows()
        print("Ingestion pipeline terminated.")

if __name__ == "__main__":
    # Ensure you have 'clip1.mp4' in this directory or provide the absolute path.
    tracker = PerceptionLayer(video_source='clip1.mp4')
    tracker.run()
