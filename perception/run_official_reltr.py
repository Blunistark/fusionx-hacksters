import torch
import cv2
import numpy as np
import sys
import os
import json
from PIL import Image
import torchvision.transforms as T

# Automatically add the current directory and parent directory to PYTHONPATH
# This ensures it finds 'models' if you put this script inside the RelTR repo.
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, '..'))

try:
    from models import build_model
except ImportError as e:
    print(f"ERROR: Official RelTR 'models' module not found. Details: {e}")
    print("Please ensure you placed this script INSIDE the cloned 'RelTR' folder.")
    sys.exit(1)

# Standard ImageNet normalization used by ResNet backbone in RelTR
transform = T.Compose([
    T.Resize(800),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

class OfficialRelTRRunner:
    """
    Wrapper for the OFFICIAL Stanford RelTR architecture.
    To be run on your high-spec College PC.
    """
    def __init__(self, checkpoint_path, num_classes=151, num_rel_classes=51, device='cuda'):
        if device == 'cuda' and not torch.cuda.is_available():
            print("WARNING: CUDA requested but not available. Falling back to CPU.")
            self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)
            
        print(f"--- RUNNING ON DEVICE: {self.device} ---")
        
        # Build args that the official build_model expects
        class Args:
            pass
        args = Args()
        args.backbone = 'resnet50'
        args.position_embedding = 'sine'
        args.hidden_dim = 256
        args.dropout = 0.1
        args.nheads = 8
        args.dim_feedforward = 2048
        args.enc_layers = 6
        args.dec_layers = 6
        args.rel_enc_layers = 1
        args.rel_dec_layers = 1
        args.num_entities = 100
        args.num_triplets = 200
        args.num_classes = num_classes
        args.num_rel_classes = num_rel_classes
        args.return_interm_layers = False
        args.lr_backbone = 1e-5
        args.masks = False
        args.dilation = False
        args.dataset = 'vg' # Visual Genome
        args.device = device

        # Instantiate official model
        self.model, self.criterion, self.postprocessors = build_model(args)
        self.model.to(self.device)
        self.model.eval()
        
        # Load the massive 300MB+ official weights
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model'])
            print("Successfully loaded official RelTR weights!")
        else:
            print(f"WARNING: Weights file not found at {checkpoint_path}")

        # VG Vocabularies
        self.CLASSES = ["__background__", "car", "person", "truck", "motorcycle", "traffic_light", "ball", "bat", "player"] # Update with your 150 classes
        self.REL_CLASSES = ["__background__", "near", "touching", "driving", "holding", "standing_next_to", "colliding_with"] # Update with your 50 relations

    def process_frame(self, frame_bgr):
        # Convert OpenCV BGR to PIL Image
        img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(img_rgb)
        
        # Transform and add batch dimension
        tensor_img = transform(pil_image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(tensor_img)
            
            # The official RelTR outputs logit tensors
            pred_logits = outputs['pred_logits'][0]
            pred_boxes = outputs['pred_boxes'][0]
            sub_logits = outputs['sub_logits'][0]
            obj_logits = outputs['obj_logits'][0]
            sub_boxes = outputs['sub_boxes'][0]
            obj_boxes = outputs['obj_boxes'][0]
            rel_logits = outputs['rel_logits'][0]
            
            prob_rel = rel_logits.softmax(-1)
            conf_rel, labels_rel = prob_rel.max(-1)
            
            # Extract high-confidence relationships
            scene_graph = {"nodes": [], "edges": []}
            
            # Use top 2 predictions for demonstration
            h, w = frame_bgr.shape[:2]
            
            valid_rels = [i for i in range(len(conf_rel)) if conf_rel[i] > 0.3 and labels_rel[i] != 0]
            
            if len(valid_rels) > 0:
                for idx in valid_rels[:2]:  # Limit to top 2 for clean UI
                    rel_class = self.REL_CLASSES[labels_rel[idx].item() % len(self.REL_CLASSES)]
                    
                    # Official RelTR outputs Subject and Object boxes separately in the relation head
                    sub_cx, sub_cy, sub_w, sub_h = sub_boxes[idx].tolist()
                    obj_cx, obj_cy, obj_w, obj_h = obj_boxes[idx].tolist()
                    
                    # Denormalize
                    sx1, sy1, sx2, sy2 = int((sub_cx-sub_w/2)*w), int((sub_cy-sub_h/2)*h), int((sub_cx+sub_w/2)*w), int((sub_cy+sub_h/2)*h)
                    ox1, oy1, ox2, oy2 = int((obj_cx-obj_w/2)*w), int((obj_cy-obj_h/2)*h), int((obj_cx+obj_w/2)*w), int((obj_cy+obj_h/2)*h)
                    
                    # Assume Subject is index 0 and Object is index 1 for this pair
                    sub_id = len(scene_graph["nodes"])
                    scene_graph["nodes"].append({
                        "id": sub_id, "label": "Subject", "bbox": [sx1, sy1, sx2, sy2], "confidence": 1.0
                    })
                    
                    obj_id = len(scene_graph["nodes"])
                    scene_graph["nodes"].append({
                        "id": obj_id, "label": "Object", "bbox": [ox1, oy1, ox2, oy2], "confidence": 1.0
                    })
                    
                    scene_graph["edges"].append({
                        "source": sub_id, "target": obj_id, "predicate": rel_class, "confidence": conf_rel[idx].item()
                    })
                    
            print(f"Extracted {len(scene_graph['edges'])} relations from Official Model!")
            return scene_graph

if __name__ == "__main__":
    print("Official RelTR Runner Ready.")
    import sys
    
    # Path to your downloaded weights
    weights_file = 'checkpoint0149.pth'
    if not os.path.exists(weights_file):
        weights_file = os.path.join('ckp', 'checkpoint0149.pth')
    
    if not os.path.exists(weights_file):
        print(f"Error: checkpoint0149.pth not found in root or ckp/ folder.")
        print("Please ensure you downloaded it from the official RelTR repo.")
        sys.exit(1)
        
    runner = OfficialRelTRRunner(weights_file)
    
    # Read a test image (create a black dummy frame if none provided)
    if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
        frame = cv2.imread(sys.argv[1])
        print(f"Processing image: {sys.argv[1]}")
    else:
        print("No image provided. Testing with a dummy black frame...")
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
    # Run inference
    graph = runner.process_frame(frame)
    print("\n--- SCENE GRAPH OUTPUT ---")
    print(json.dumps(graph, indent=2) if 'json' in sys.modules else graph)
    print("--------------------------")
    print("Test successful!")
