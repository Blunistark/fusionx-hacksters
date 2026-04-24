import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import logging
import time
import numpy as np
from reltr_detector import RelTR

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import os
import json
import cv2

class VisualGenomeDataset(Dataset):
    """
    Actual Scene Graph Dataset loader.
    Expects a directory of images and a JSON file containing annotations.
    JSON structure should follow standard Scene Graph format:
    [
      {
        "image_id": "123",
        "file_name": "image_123.jpg",
        "objects": [{"box": [cx, cy, w, h], "class_id": 5}],
        "relations": [{"source": 0, "target": 1, "predicate": 12}]
      }
    ]
    """
    def __init__(self, img_dir: str, annotation_json: str, max_queries: int = 10):
        self.img_dir = img_dir
        self.max_queries = max_queries
        
        logger.info(f"Loading dataset annotations from {annotation_json}...")
        if os.path.exists(annotation_json):
            with open(annotation_json, 'r') as f:
                self.data = json.load(f)
            logger.info(f"Loaded {len(self.data)} actual training samples.")
        else:
            logger.warning(f"Annotation file {annotation_json} not found. Creating empty dataset.")
            self.data = []

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        img_path = os.path.join(self.img_dir, item['file_name'])
        
        # Load and preprocess image
        if os.path.exists(img_path):
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (800, 800))
            img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        else:
            # Fallback black image if missing
            img_tensor = torch.zeros(3, 800, 800)
            
        # Parse ground truth
        gt_classes = torch.zeros(self.max_queries, dtype=torch.long)
        gt_boxes = torch.zeros((self.max_queries, 4), dtype=torch.float32)
        gt_relations = torch.zeros(self.max_queries, dtype=torch.long)
        
        for i, obj in enumerate(item.get('objects', [])[:self.max_queries]):
            gt_classes[i] = obj['class_id']
            gt_boxes[i] = torch.tensor(obj['box'])
            
        for i, rel in enumerate(item.get('relations', [])[:self.max_queries]):
            # Simplified relation mapping for 1D target loss
            gt_relations[i] = rel['predicate']
            
        return img_tensor, gt_classes, gt_boxes, gt_relations


def train_reltr(epochs=5, batch_size=4, lr=1e-4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Starting RelTR training on {device}...")
    
    # 1. Initialize Dataset & Loader
    # Specify your actual data paths here:
    data_dir = "d:/fusionx-hacksters/data/images"
    annotations = "d:/fusionx-hacksters/data/scene_graphs.json"
    dataset = VisualGenomeDataset(img_dir=data_dir, annotation_json=annotations)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # 2. Initialize Model
    model = RelTR(num_classes=151, num_rel_classes=51).to(device)
    model.train()
    
    # 3. Optimizer & Loss Functions
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    
    criterion_class = nn.CrossEntropyLoss()
    criterion_bbox = nn.MSELoss() # Simplified, usually L1 + GIoU for DETR architectures
    criterion_rel = nn.CrossEntropyLoss()
    
    # 4. Training Loop
    for epoch in range(epochs):
        epoch_loss = 0.0
        start_time = time.time()
        
        for batch_idx, (images, gt_classes, gt_boxes, gt_relations) in enumerate(dataloader):
            # Move to device
            images = images.to(device)
            gt_classes = gt_classes.to(device)
            gt_boxes = gt_boxes.to(device)
            gt_relations = gt_relations.to(device)
            
            # Forward Pass
            optimizer.zero_grad()
            outputs = model(images)
            
            # Extract predictions
            pred_logits = outputs['pred_logits'] # (B, 10, num_classes+1)
            pred_boxes = outputs['pred_boxes']   # (B, 10, 4)
            pred_rel = outputs['pred_rel']       # (B, 10, num_rel_classes+1)
            
            # Reshape for loss functions
            # CrossEntropy expects (N, C) and targets (N)
            B, Q = pred_logits.shape[:2]
            
            loss_cls = criterion_class(pred_logits.view(B * Q, -1), gt_classes.view(-1))
            loss_bbox = criterion_bbox(pred_boxes, gt_boxes)
            loss_rel = criterion_rel(pred_rel.view(B * Q, -1), gt_relations.view(-1))
            
            # Total Loss
            total_loss = loss_cls + (5.0 * loss_bbox) + loss_rel
            
            # Backward Pass & Optimize
            total_loss.backward()
            optimizer.step()
            
            epoch_loss += total_loss.item()
            
            if batch_idx % 10 == 0:
                logger.info(f"Epoch [{epoch+1}/{epochs}] | Batch [{batch_idx}/{len(dataloader)}] | "
                            f"Loss: {total_loss.item():.4f} (Cls: {loss_cls.item():.2f}, "
                            f"Box: {loss_bbox.item():.2f}, Rel: {loss_rel.item():.2f})")
                
        epoch_duration = time.time() - start_time
        logger.info(f"=== Epoch {epoch+1} Completed | Avg Loss: {epoch_loss/len(dataloader):.4f} | "
                    f"Time: {epoch_duration:.2f}s ===")
        
    # Save the model
    torch.save(model.state_dict(), "reltr_hackathon_checkpoint.pth")
    logger.info("Training complete. Model weights saved to 'reltr_hackathon_checkpoint.pth'")


if __name__ == "__main__":
    train_reltr()
