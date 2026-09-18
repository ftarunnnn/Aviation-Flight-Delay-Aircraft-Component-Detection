"""
Phase 1: Data Collection Script
Aviation - Flight Delay & Aircraft Component Detection
Generates raw flight operational data and aircraft component inspection images/metadata.
"""

import os
import json
import random
import datetime
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
IMAGE_DIR = os.path.join(RAW_DATA_DIR, "inspection_images")

os.makedirs(RAW_DATA_DIR, exist_ok=True)
os.makedirs(IMAGE_DIR, exist_ok=True)

print("Starting Phase 1: Data Collection...")

# 1. Generate Flight Operational Dataset
AIRLINES = ["SkyWings", "AeroGlobal", "JetExpress", "OceanicAir", "TransFlight", "NimbusAero"]
AIRPORTS = ["JFK", "LAX", "ORD", "ATL", "DFW", "LHR", "CDG", "HND", "DXB", "FRA"]
WEATHER_CONDITIONS = ["Clear", "Overcast", "Rain", "Thunderstorm", "Snow", "Fog", "High Winds"]

num_flights = 1200
start_date = datetime.datetime(2026, 1, 1, 6, 0)

flights_data = []

for i in range(num_flights):
    flight_id = f"FL-{1000 + i}"
    airline = random.choice(AIRLINES)
    flight_num = f"{airline[:2].upper()}{random.randint(100, 999)}"
    origin = random.choice(AIRPORTS)
    dest = random.choice([a for a in AIRPORTS if a != origin])
    
    # Departure time sampling
    dep_delta = datetime.timedelta(minutes=random.randint(0, 1440 * 30))
    sched_dep = start_date + dep_delta
    flight_duration = datetime.timedelta(minutes=random.randint(45, 420))
    sched_arr = sched_dep + flight_duration
    
    weather = random.choice(WEATHER_CONDITIONS)
    temp_c = round(random.uniform(-10.0, 38.0), 1)
    wind_kts = round(random.uniform(2.0, 45.0), 1)
    visibility_km = round(random.uniform(0.5, 10.0), 1)
    traffic_index = round(random.uniform(1.0, 10.0), 2)  # 1 (low) to 10 (congested)
    prior_delay_min = max(0, int(np.random.exponential(scale=15)))
    aircraft_age = round(random.uniform(0.5, 22.0), 1)
    maint_overdue = 1 if (aircraft_age > 15.0 and random.random() < 0.25) else 0
    
    # Calculate target actual departure delay with realistic physical relations
    weather_impact = {
        "Clear": 0, "Overcast": 3, "Rain": 12, "Thunderstorm": 45, 
        "Snow": 35, "Fog": 25, "High Winds": 30
    }[weather]
    
    base_delay = (
        weather_impact * random.uniform(0.7, 1.3) +
        (traffic_index * 3.5) +
        (prior_delay_min * 0.6) +
        (maint_overdue * 20.0) +
        np.random.normal(loc=0, scale=10)
    )
    actual_dep_delay = max(0, int(round(base_delay)))
    
    flights_data.append({
        "flight_id": flight_id,
        "airline": airline,
        "flight_number": flight_num,
        "origin_airport": origin,
        "destination_airport": dest,
        "scheduled_departure": sched_dep.strftime("%Y-%m-%d %H:%M:%S"),
        "scheduled_arrival": sched_arr.strftime("%Y-%m-%d %H:%M:%S"),
        "weather_condition": weather,
        "temperature_c": temp_c,
        "wind_speed_kts": wind_kts,
        "visibility_km": visibility_km,
        "airport_traffic_index": traffic_index,
        "prior_leg_delay_min": prior_delay_min,
        "aircraft_age_years": aircraft_age,
        "maintenance_overdue_flag": maint_overdue,
        "actual_departure_delay_min": actual_dep_delay
    })

flights_df = pd.DataFrame(flights_data)
csv_path = os.path.join(RAW_DATA_DIR, "flights_raw.csv")
flights_df.to_csv(csv_path, index=False)
print(f"Generated {len(flights_df)} flight records at: {csv_path}")

# 2. Generate Synthetic Aircraft Inspection Images & Annotations
DEFECT_TYPES = ["Rivet Corrosion", "Fuselage Crack", "Turbine Blade Erosion", "Wing Surface Dent", "Composite Delamination"]
COMPONENTS = ["Fuselage Panel", "Turbine Blade", "Wing Surface", "Landing Gear Structure", "Engine Cowling"]

images_metadata = []

def create_synthetic_inspection_image(filename, component, defects):
    # Base metallic image background
    img = Image.new('RGB', (640, 640), color=(180, 185, 190))
    draw = ImageDraw.Draw(img)
    
    # Add metallic texture/noise
    arr = np.array(img)
    noise = np.random.randint(-15, 15, arr.shape, dtype=np.int16)
    arr = np.clip(arr.astype(np.int16) + noise, 0, 255).astype(uint8:=np.uint8)
    img = Image.fromarray(arr)
    draw = ImageDraw.Draw(img)
    
    # Draw structural component panel lines
    draw.line([(0, 320), (640, 320)], fill=(120, 125, 130), width=3)
    draw.line([(320, 0), (320, 640)], fill=(120, 125, 130), width=3)
    
    # Add rivets around borders
    for x in range(40, 640, 80):
        draw.ellipse([x-6, 40-6, x+6, 40+6], fill=(100, 105, 110), outline=(60, 65, 70))
        draw.ellipse([x-6, 600-6, x+6, 600+6], fill=(100, 105, 110), outline=(60, 65, 70))

    # Draw synthetic defects
    for defect in defects:
        bbox = defect["bbox"]  # [x, y, w, h]
        d_type = defect["defect_type"]
        x, y, w, h = bbox
        
        if d_type == "Rivet Corrosion":
            draw.ellipse([x, y, x+w, y+h], fill=(160, 80, 40), outline=(100, 40, 20))
        elif d_type == "Fuselage Crack":
            draw.line([(x, y), (x+w//2, y+h//2), (x+w, y+h)], fill=(30, 30, 30), width=4)
        elif d_type == "Turbine Blade Erosion":
            draw.rectangle([x, y, x+w, y+h], fill=(140, 130, 110), outline=(80, 70, 50))
        elif d_type == "Wing Surface Dent":
            draw.ellipse([x, y, x+w, y+h], fill=(150, 155, 160), outline=(90, 95, 100))
        elif d_type == "Composite Delamination":
            draw.polygon([(x, y+h), (x+w//2, y), (x+w, y+h)], fill=(70, 75, 80), outline=(40, 45, 50))

    img_path = os.path.join(IMAGE_DIR, filename)
    img.save(img_path)
    return img_path

# Create 25 sample inspection images with ground truth annotations
for idx in range(1, 26):
    img_name = f"inspection_img_{idx:03d}.jpg"
    comp = random.choice(COMPONENTS)
    num_defects = random.randint(1, 3)
    defects = []
    
    for _ in range(num_defects):
        d_type = random.choice(DEFECT_TYPES)
        w = random.randint(40, 120)
        h = random.randint(40, 120)
        x = random.randint(50, 640 - w - 50)
        y = random.randint(50, 640 - h - 50)
        severity = random.choice(["Minor", "Moderate", "Critical"])
        
        defects.append({
            "defect_type": d_type,
            "bbox": [x, y, w, h],  # [x_min, y_min, width, height]
            "severity": severity,
            "confidence_ground_truth": round(random.uniform(0.88, 0.99), 2)
        })
    
    create_synthetic_inspection_image(img_name, comp, defects)
    
    images_metadata.append({
        "image_id": f"IMG-{idx:03d}",
        "filename": img_name,
        "component": comp,
        "defects": defects,
        "inspection_date": (datetime.datetime(2026, 2, 1) + datetime.timedelta(days=idx)).strftime("%Y-%m-%d"),
        "resolution": [640, 640]
    })

json_path = os.path.join(RAW_DATA_DIR, "inspection_annotations_raw.json")
with open(json_path, 'w') as f:
    json.dump(images_metadata, f, indent=2)

print(f"Generated {len(images_metadata)} inspection images and raw annotations at: {json_path}")
print("Phase 1 Data Collection Completed Successfully!")
