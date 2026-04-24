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

class MockSceneGraphDataset(Dataset):
    """
    A mock PyTorch dataset mimicking Visual Genome for Scene Graph Generation.
    Returns random image tensors and mock ground-truth graphs.
    """
    def __init__(self, num_samples=100, max_queries=10):
        self.num_samples = num_samples
        self.max_queries = max_queries

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # 1. Generate random image tensor (3, 800, 800)
        img = torch.rand(3, 800, 800)
        
        # 2. Generate random ground truth for 10 queries
        # Classes: 0-150 (random)
        gt_classes = torch.randint(0, 151, (self.max_queries,))
        
        # Bounding boxes: [cx, cy, w, h] normalized between 0 and 1
        gt_boxes = torch.rand(self.max_queries, 4)
        
        # Relations: 0-50 (random)
        gt_relations = torch.randint(0, 51, (self.max_queries,))
        
        return img, gt_classes, gt_boxes, gt_relations


def train_reltr(epochs=5, batch_size=4, lr=1e-4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Starting RelTR training on {device}...")
    
    # 1. Initialize Dataset & Loader
    dataset = MockSceneGraphDataset(num_samples=200) # Small dataset for hackathon demo
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
