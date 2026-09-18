"""
Phase 7: DL Image Processing & Augmentation Script
Aviation - Flight Delay & Aircraft Component Detection
Performs augmentation (rotation, contrast, flips) and outputs normalized YOLO format annotations.
"""

import os
import json
import numpy as np
from PIL import Image, ImageEnhance, ImageOps

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
AUGMENTED_DIR = os.path.join(PROCESSED_DATA_DIR, "inspection_augmented")
LABELS_DIR = os.path.join(AUGMENTED_DIR, "labels")
IMAGES_DIR = os.path.join(AUGMENTED_DIR, "images")

os.makedirs(LABELS_DIR, exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)

print("Starting Phase 7: DL Image Processing...")

# Defect class mapping
DEFECT_CLASS_MAP = {
    "Rivet Corrosion": 0,
    "Fuselage Crack": 1,
    "Turbine Blade Erosion": 2,
    "Wing Surface Dent": 3,
    "Composite Delamination": 4
}

# Load preprocessed annotations
with open(os.path.join(PROCESSED_DATA_DIR, "inspection_annotations_clean.json"), 'r', encoding='utf-8') as f:
    clean_meta = json.load(f)

augmented_records = []

for item in clean_meta:
    orig_img_path = os.path.join(PROCESSED_DATA_DIR, "inspection_images_224", item["filename"])
    if not os.path.exists(orig_img_path):
        continue
    
    base_img = Image.open(orig_img_path)
    base_name = item["original_filename"].split('.')[0]
    
    # 1. Save Original in Augmented collection
    out_img_name_orig = f"{base_name}_orig.jpg"
    base_img.save(os.path.join(IMAGES_DIR, out_img_name_orig))
    
    # Write YOLO format label file
    label_filename_orig = f"{base_name}_orig.txt"
    with open(os.path.join(LABELS_DIR, label_filename_orig), 'w') as lf:
        for defect in item["defects"]:
            cid = DEFECT_CLASS_MAP.get(defect["defect_type"], 0)
            xc, yc, w, h = defect["yolo_norm_bbox"]
            lf.write(f"{cid} {xc:.4f} {yc:.4f} {w:.4f} {h:.4f}\n")

    # 2. Augmentation 1: Horizontal Flip
    flipped_img = ImageOps.mirror(base_img)
    out_img_name_flip = f"{base_name}_flip.jpg"
    flipped_img.save(os.path.join(IMAGES_DIR, out_img_name_flip))
    
    label_filename_flip = f"{base_name}_flip.txt"
    with open(os.path.join(LABELS_DIR, label_filename_flip), 'w') as lf:
        for defect in item["defects"]:
            cid = DEFECT_CLASS_MAP.get(defect["defect_type"], 0)
            xc, yc, w, h = defect["yolo_norm_bbox"]
            # Flipped x center
            xc_flip = round(1.0 - xc, 4)
            lf.write(f"{cid} {xc_flip:.4f} {yc:.4f} {w:.4f} {h:.4f}\n")

    # 3. Augmentation 2: Brightness & Contrast Adjustment
    enhancer = ImageEnhance.Brightness(base_img)
    bright_img = enhancer.enhance(1.2)
    enhancer = ImageEnhance.Contrast(bright_img)
    bright_contrast_img = enhancer.enhance(1.3)
    out_img_name_bright = f"{base_name}_bright.jpg"
    bright_contrast_img.save(os.path.join(IMAGES_DIR, out_img_name_bright))
    
    label_filename_bright = f"{base_name}_bright.txt"
    with open(os.path.join(LABELS_DIR, label_filename_bright), 'w') as lf:
        for defect in item["defects"]:
            cid = DEFECT_CLASS_MAP.get(defect["defect_type"], 0)
            xc, yc, w, h = defect["yolo_norm_bbox"]
            lf.write(f"{cid} {xc:.4f} {yc:.4f} {w:.4f} {h:.4f}\n")

    augmented_records.append({
        "base_image_id": item["image_id"],
        "augmented_files": [out_img_name_orig, out_img_name_flip, out_img_name_bright],
        "component": item["component"],
        "num_defects": len(item["defects"])
    })

# Save metadata index
yolo_meta_path = os.path.join(PROCESSED_DATA_DIR, "yolo_dataset_metadata.json")
with open(yolo_meta_path, 'w', encoding='utf-8') as f:
    json.dump({
        "class_mapping": DEFECT_CLASS_MAP,
        "total_augmented_images": len(augmented_records) * 3,
        "records": augmented_records
    }, f, indent=2)

print(f"Generated {len(augmented_records) * 3} augmented inspection images and YOLO labels.")
print(f"Dataset metadata saved to: {yolo_meta_path}")
print("Phase 7 DL Image Processing Completed Successfully!")
