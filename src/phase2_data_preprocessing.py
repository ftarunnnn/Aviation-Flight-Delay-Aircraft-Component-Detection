"""
Phase 2: Data Preprocessing Script
Aviation - Flight Delay & Aircraft Component Detection
Cleans raw flight data, handles missing values, encodes categoricals, and normalizes inspection image metadata.
"""

import os
import json
import numpy as np
import pandas as pd
from PIL import Image

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
PROCESSED_IMG_DIR = os.path.join(PROCESSED_DATA_DIR, "inspection_images_224")

os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(PROCESSED_IMG_DIR, exist_ok=True)

print("Starting Phase 2: Data Preprocessing...")

# 1. Flight Data Cleaning & Preprocessing
raw_csv_path = os.path.join(RAW_DATA_DIR, "flights_raw.csv")
df = pd.read_csv(raw_csv_path)
print(f"Loaded raw dataset with {len(df)} rows.")

# Check & Handle missing values
df['prior_leg_delay_min'] = df['prior_leg_delay_min'].fillna(0)
df['temperature_c'] = df['temperature_c'].fillna(df['temperature_c'].median())
df['wind_speed_kts'] = df['wind_speed_kts'].fillna(df['wind_speed_kts'].median())
df['visibility_km'] = df['visibility_km'].fillna(df['visibility_km'].mean())

# Convert scheduled datetime strings to datetime objects
df['scheduled_departure'] = pd.to_datetime(df['scheduled_departure'])
df['scheduled_arrival'] = pd.to_datetime(df['scheduled_arrival'])

# Categorical Encoding
airline_mapping = {airline: idx for idx, airline in enumerate(df['airline'].unique())}
weather_mapping = {weather: idx for idx, weather in enumerate(df['weather_condition'].unique())}

df['airline_code'] = df['airline'].map(airline_mapping)
df['weather_code'] = df['weather_condition'].map(weather_mapping)

# Numerical Scaling & Normalization (Min-Max scaling sample stats saved)
df['temp_norm'] = (df['temperature_c'] - df['temperature_c'].min()) / (df['temperature_c'].max() - df['temperature_c'].min())
df['wind_norm'] = (df['wind_speed_kts'] - df['wind_speed_kts'].min()) / (df['wind_speed_kts'].max() - df['wind_speed_kts'].min())
df['traffic_norm'] = (df['airport_traffic_index'] - df['airport_traffic_index'].min()) / (df['airport_traffic_index'].max() - df['airport_traffic_index'].min())

# Create Delay Risk Classification Target (0: On-Time/Minor <15m, 1: Moderate 15-45m, 2: Severe >45m)
def categorize_delay(minutes):
    if minutes < 15:
        return 0  # Low Risk / On-Time
    elif minutes <= 45:
        return 1  # Moderate Delay Risk
    else:
        return 2  # High Delay Risk

df['delay_risk_class'] = df['actual_departure_delay_min'].apply(categorize_delay)

clean_csv_path = os.path.join(PROCESSED_DATA_DIR, "flights_clean.csv")
df.to_csv(clean_csv_path, index=False)
print(f"Cleaned flight data saved to: {clean_csv_path}")

# 2. Image Preprocessing & Annotation Normalization
raw_json_path = os.path.join(RAW_DATA_DIR, "inspection_annotations_raw.json")
with open(raw_json_path, 'r') as f:
    raw_img_meta = json.load(f)

clean_img_meta = []
TARGET_SIZE = (224, 224)
ORIG_SIZE = (640, 640)

scale_x = TARGET_SIZE[0] / ORIG_SIZE[0]
scale_y = TARGET_SIZE[1] / ORIG_SIZE[1]

for item in raw_img_meta:
    raw_img_path = os.path.join(RAW_DATA_DIR, "inspection_images", item["filename"])
    if os.path.exists(raw_img_path):
        img = Image.open(raw_img_path)
        img_resized = img.resize(TARGET_SIZE, Image.Resampling.LANCZOS)
        
        proc_img_filename = f"resized_{item['filename']}"
        proc_img_path = os.path.join(PROCESSED_IMG_DIR, proc_img_filename)
        img_resized.save(proc_img_path)
        
        # Rescale Bounding Boxes & compute YOLO Normalized format [x_center, y_center, width, height]
        processed_defects = []
        for defect in item["defects"]:
            x, y, w, h = defect["bbox"]
            # Rescaled absolute bounding box
            rx = round(x * scale_x, 2)
            ry = round(y * scale_y, 2)
            rw = round(w * scale_x, 2)
            rh = round(h * scale_y, 2)
            
            # YOLO normalized bbox relative to 640x640 canvas (0.0 to 1.0)
            x_center_norm = round((x + w / 2.0) / ORIG_SIZE[0], 4)
            y_center_norm = round((y + h / 2.0) / ORIG_SIZE[1], 4)
            width_norm = round(w / ORIG_SIZE[0], 4)
            height_norm = round(h / ORIG_SIZE[1], 4)
            
            processed_defects.append({
                "defect_type": defect["defect_type"],
                "severity": defect["severity"],
                "bbox_224": [rx, ry, rw, rh],
                "yolo_norm_bbox": [x_center_norm, y_center_norm, width_norm, height_norm],
                "confidence_ground_truth": defect["confidence_ground_truth"]
            })
            
        clean_img_meta.append({
            "image_id": item["image_id"],
            "filename": proc_img_filename,
            "original_filename": item["filename"],
            "component": item["component"],
            "defects": processed_defects,
            "resolution": TARGET_SIZE
        })

clean_json_path = os.path.join(PROCESSED_DATA_DIR, "inspection_annotations_clean.json")
with open(clean_json_path, 'w') as f:
    json.dump(clean_img_meta, f, indent=2)

print(f"Processed {len(clean_img_meta)} image records saved to: {clean_json_path}")
print("Phase 2 Data Preprocessing Completed Successfully!")
