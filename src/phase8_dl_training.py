"""
Phase 8: DL Model Training Script
Aviation - Flight Delay & Aircraft Component Detection
Trains CNN Defect Classifier & builds YOLO object detection configuration pipeline.
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(MODELS_DIR, exist_ok=True)

print("Starting Phase 8: DL Model Training...")

# Set PyTorch random seed
torch.manual_seed(42)

# 1. Define CNN Architecture for Component Defect Classification
class AircraftDefectCNN(nn.Module):
    def __init__(self, num_classes=5):
        super(AircraftDefectCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 112x112
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 56x56
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 28x28
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 28 * 28, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# Dataset loader helper
class InspectionDataset(Dataset):
    def __init__(self, metadata_path):
        with open(metadata_path, 'r', encoding='utf-8') as f:
            self.meta = json.load(f)
        self.class_map = {
            "Rivet Corrosion": 0, "Fuselage Crack": 1,
            "Turbine Blade Erosion": 2, "Wing Surface Dent": 3,
            "Composite Delamination": 4
        }
        self.samples = []
        for item in self.meta:
            img_path = os.path.join(PROCESSED_DATA_DIR, "inspection_images_224", item["filename"])
            if os.path.exists(img_path) and len(item["defects"]) > 0:
                # Primary defect class label
                primary_label = self.class_map[item["defects"][0]["defect_type"]]
                self.samples.append((img_path, primary_label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        arr = np.array(img, dtype=np.float32) / 255.0  # Normalize to [0, 1]
        arr = np.transpose(arr, (2, 0, 1))  # Convert to C, H, W
        return torch.tensor(arr), torch.tensor(label, dtype=torch.long)

# Train CNN Model
dataset = InspectionDataset(os.path.join(PROCESSED_DATA_DIR, "inspection_annotations_clean.json"))
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

model = AircraftDefectCNN(num_classes=5)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print(f"Training CNN on {len(dataset)} inspection samples over 15 epochs...")
model.train()
for epoch in range(1, 16):
    running_loss = 0.0
    for imgs, labels in dataloader:
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)
    
    epoch_loss = running_loss / len(dataset)
    if epoch % 5 == 0 or epoch == 1:
        print(f" Epoch [{epoch:02d}/15] - Loss: {epoch_loss:.4f}")

# Save CNN Weights
cnn_path = os.path.join(MODELS_DIR, "defect_classifier_cnn.pt")
torch.save(model.state_dict(), cnn_path)
print(f"Trained PyTorch CNN model saved at: {cnn_path}")

# 2. Save YOLO Model Configuration Metadata
yolo_config = {
    "architecture": "YOLOv8-Custom-Aviation",
    "input_resolution": [640, 640],
    "num_classes": 5,
    "classes": ["Rivet Corrosion", "Fuselage Crack", "Turbine Blade Erosion", "Wing Surface Dent", "Composite Delamination"],
    "anchors": [[10, 13], [16, 30], [33, 23], [30, 61], [62, 45], [59, 119]],
    "iou_threshold": 0.45,
    "confidence_threshold": 0.50
}
yolo_cfg_path = os.path.join(MODELS_DIR, "yolo_detector_config.json")
with open(yolo_cfg_path, 'w', encoding='utf-8') as f:
    json.dump(yolo_config, f, indent=2)

print(f"YOLO detector configuration persisted at: {yolo_cfg_path}")
print("Phase 8 DL Model Training Completed Successfully!")
