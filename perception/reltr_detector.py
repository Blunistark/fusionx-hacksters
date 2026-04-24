import torch
import torch.nn as nn
import cv2
import numpy as np
import logging
from typing import List, Dict, Tuple

logger = logging.getLogger(__name__)

class RelTR(nn.Module):
    """
    Mock/Skeleton Implementation of the Relation Transformer (RelTR).
    In a full production environment, this would load the pretrained ResNet backbone
    and the Transformer encoder-decoder layers to predict (Subject, Predicate, Object).
    """
    def __init__(self, num_classes=151, num_rel_classes=51):
        super().__init__()
        # 1. Backbone (e.g., ResNet-50)
        # self.backbone = resnet50()
        
        # 2. Transformer
        # self.transformer = Transformer(...)
        
        # 3. Prediction Heads
        self.class_embed = nn.Linear(256, num_classes + 1)
        self.bbox_embed = nn.Linear(256, 4) # MLP for bounding boxes
        self.rel_embed = nn.Linear(256, num_rel_classes + 1)
        # 4. Projection for dummy backbone
        self.proj = nn.Linear(3, 256)
        
        logger.info("RelTR Model Architecture Initialized")

    def forward(self, images):
        """
        Functional forward pass for training.
        images: (batch_size, 3, H, W)
        Returns class logits, bbox coords, and relation logits for 10 queries.
        """
        batch_size = images.shape[0]
        
        # Dummy feature extraction (global average pool)
        x = torch.mean(images, dim=(2, 3)) # (batch_size, 3)
        x = self.proj(x).unsqueeze(1) # (batch_size, 1, 256)
        
        # Simulate 10 transformer object queries
        x = x.repeat(1, 10, 1) # (batch_size, 10, 256)
        
        outputs_class = self.class_embed(x)
        outputs_coord = torch.sigmoid(self.bbox_embed(x)) # Bboxes in [0, 1]
        outputs_rel = self.rel_embed(x)
        
        return {'pred_logits': outputs_class, 'pred_boxes': outputs_coord, 'pred_rel': outputs_rel}

class RelTRSceneGraphGenerator:
    """
    Wraps the RelTR PyTorch model to process OpenCV video frames
    and generate dynamic scene graphs (Nodes & Edges).
    """
    def __init__(self, weights_path: str = None, device: str = None):
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Loading RelTR SGG on device: {self.device}")
        
        self.model = RelTR().to(self.device)
        self.model.eval()
        
        if weights_path:
            try:
                self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
                logger.info(f"Loaded weights from {weights_path}")
            except Exception as e:
                logger.warning(f"Could not load weights: {e}. Using untrained weights.")
            
        # Example Vocabularies (Visual Genome format)
        self.object_classes = ["__background__", "car", "person", "truck", "motorcycle", "traffic_light", "ball", "bat", "player"]
        self.predicate_classes = ["__background__", "near", "touching", "driving", "holding", "standing_next_to", "colliding_with"]
        
    def preprocess_frame(self, frame: np.ndarray) -> torch.Tensor:
        """Convert BGR OpenCV frame to standard PyTorch format"""
        # Resize, normalize, to tensor
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (800, 800))
        img = img.astype(np.float32) / 255.0
        
        # HWC to CHW
        tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
        return tensor.to(self.device)

    def generate_scene_graph(self, frame: np.ndarray, confidence_threshold: float = 0.5) -> Dict:
        """
        Process a frame through RelTR and output the Scene Graph.
        Returns a dictionary containing Nodes and Edges.
        """
        tensor_img = self.preprocess_frame(frame)
        
        with torch.no_grad():
            outputs = self.model(tensor_img)
            
            # Parse predictions
            pred_logits = outputs['pred_logits'][0] # (10, num_classes)
            pred_boxes = outputs['pred_boxes'][0]   # (10, 4)
            pred_rel = outputs['pred_rel'][0]       # (10, num_rel_classes)
            
            # Apply softmax to get probabilities
            prob_class = torch.softmax(pred_logits, dim=-1)
            prob_rel = torch.softmax(pred_rel, dim=-1)
            
            # Get max probabilities and corresponding classes
            conf_class, labels_class = prob_class.max(-1)
            conf_rel, labels_rel = prob_rel.max(-1)
            
            h, w = frame.shape[:2]
            
            nodes = []
            edges = []
            
            # Since our dummy dataset generates random relationships for random bounding boxes,
            # we'll extract the top 2 highest confidence nodes to form an edge for demonstration.
            # In a real SGG model, subject/object index pairing is predicted directly.
            
            valid_indices = [i for i in range(len(conf_class)) if conf_class[i] > confidence_threshold and labels_class[i] != 0]
            
            # Fallback if no valid predictions (especially with untrained/mock weights)
            if len(valid_indices) < 2:
                # Mock subject: Car
                subj_box = [100, 150, 300, 250]
                subj_class = "car"
                
                # Mock object: Person
                obj_box = [280, 200, 320, 280]
                obj_class = "person"
                
                # Mock Predicate: colliding_with
                predicate = "colliding_with"
                
                nodes = [
                    {"id": 0, "label": subj_class, "bbox": subj_box, "confidence": 0.92},
                    {"id": 1, "label": obj_class, "bbox": obj_box, "confidence": 0.88}
                ]
                edges = [{"source": 0, "target": 1, "predicate": predicate, "confidence": 0.75}]
            else:
                for idx in valid_indices:
                    label_idx = labels_class[idx].item()
                    label_name = self.object_classes[label_idx % len(self.object_classes)]
                    
                    # Convert [cx, cy, w, h] to [x1, y1, x2, y2] and scale to frame dims
                    cx, cy, bw, bh = pred_boxes[idx].tolist()
                    x1 = int((cx - bw/2) * w)
                    y1 = int((cy - bh/2) * h)
                    x2 = int((cx + bw/2) * w)
                    y2 = int((cy + bh/2) * h)
                    
                    nodes.append({
                        "id": idx,
                        "label": label_name,
                        "bbox": [x1, y1, x2, y2],
                        "confidence": conf_class[idx].item()
                    })
                
                # Create an edge between the first two valid nodes
                if len(nodes) >= 2:
                    rel_idx = labels_rel[valid_indices[0]].item()
                    predicate = self.predicate_classes[rel_idx % len(self.predicate_classes)]
                    edges.append({
                        "source": nodes[0]['id'],
                        "target": nodes[1]['id'],
                        "predicate": predicate,
                        "confidence": conf_rel[valid_indices[0]].item()
                    })
            
            scene_graph = {
                "nodes": nodes,
                "edges": edges
            }
            
            return scene_graph

    def draw_scene_graph(self, frame: np.ndarray, scene_graph: Dict) -> np.ndarray:
        """Draw the nodes and edges on the frame"""
        annotated_frame = frame.copy()
        
        nodes = {n['id']: n for n in scene_graph['nodes']}
        
        # Draw Nodes (Bounding Boxes)
        for node in scene_graph['nodes']:
            x1, y1, x2, y2 = node['bbox']
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated_frame, node['label'], (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        
        # Draw Edges (Relationships)
        for edge in scene_graph['edges']:
            src = nodes[edge['source']]
            tgt = nodes[edge['target']]
            
            # Draw line between centers
            src_center = ((src['bbox'][0] + src['bbox'][2]) // 2, (src['bbox'][1] + src['bbox'][3]) // 2)
            tgt_center = ((tgt['bbox'][0] + tgt['bbox'][2]) // 2, (tgt['bbox'][1] + tgt['bbox'][3]) // 2)
            
            cv2.line(annotated_frame, src_center, tgt_center, (0, 0, 255), 2)
            
            # Put predicate text
            mid_pt = ((src_center[0] + tgt_center[0]) // 2, (src_center[1] + tgt_center[1]) // 2)
            cv2.putText(annotated_frame, edge['predicate'], mid_pt,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
        return annotated_frame

if __name__ == "__main__":
    # Test the standalone RelTR module
    logger.setLevel(logging.DEBUG)
    
    # 1. Initialize Transformer
    sgg = RelTRSceneGraphGenerator()
    
    # 2. Create dummy frame
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # 3. Generate Graph
    graph = sgg.generate_scene_graph(dummy_frame)
    print("--- GENERATED SCENE GRAPH ---")
    print(graph)
    
    # 4. Render
    output = sgg.draw_scene_graph(dummy_frame, graph)
    print("\nScene Graph drawn successfully onto frame array.")
