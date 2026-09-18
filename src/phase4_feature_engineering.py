"""
Phase 4: Feature Engineering Script
Aviation - Flight Delay & Aircraft Component Detection
Engineers domain-specific temporal, weather severity, congestion risk, and aircraft operational features.
"""

import os
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data", "processed")

print("Starting Phase 4: Feature Engineering...")

clean_csv_path = os.path.join(PROCESSED_DATA_DIR, "flights_clean.csv")
df = pd.read_csv(clean_csv_path)

# Convert datetimes
df['scheduled_departure'] = pd.to_datetime(df['scheduled_departure'])
df['scheduled_arrival'] = pd.to_datetime(df['scheduled_arrival'])

# 1. Temporal Feature Extraction
df['dep_hour'] = df['scheduled_departure'].dt.hour
df['dep_day_of_week'] = df['scheduled_departure'].dt.dayofweek
df['is_weekend'] = df['dep_day_of_week'].apply(lambda d: 1 if d in [5, 6] else 0)
df['is_peak_hours'] = df['dep_hour'].apply(lambda h: 1 if (7 <= h <= 10 or 16 <= h <= 20) else 0)

# 2. Weather Severity Index (Domain Mapping)
weather_severity_map = {
    "Clear": 1.0,
    "Overcast": 2.5,
    "Rain": 5.0,
    "Fog": 6.5,
    "High Winds": 7.5,
    "Snow": 8.5,
    "Thunderstorm": 10.0
}
df['weather_severity_score'] = df['weather_condition'].map(weather_severity_map)

# 3. Airport Congestion & Environmental Composite Interaction
df['congestion_risk_score'] = round(df['airport_traffic_index'] * df['weather_severity_score'], 2)

# 4. Aircraft Health & Wear Indicator
df['aircraft_wear_index'] = round(df['aircraft_age_years'] * (1.0 + (df['maintenance_overdue_flag'] * 1.5)), 2)

# 5. Cascading Delay Propagation Metric
df['prior_delay_ratio'] = round(df['prior_leg_delay_min'] / 45.0, 3)  # Relative to standard 45-min turnaround window

# 6. Route Estimated Duration (Minutes)
df['scheduled_flight_duration_min'] = (df['scheduled_arrival'] - df['scheduled_departure']).dt.total_seconds() / 60.0

engineered_csv_path = os.path.join(PROCESSED_DATA_DIR, "flights_engineered.csv")
df.to_csv(engineered_csv_path, index=False)

print(f"Engineered dataset with {df.shape[1]} features created at: {engineered_csv_path}")
print("New Features Added:")
print(" - dep_hour, dep_day_of_week, is_weekend, is_peak_hours")
print(" - weather_severity_score, congestion_risk_score")
print(" - aircraft_wear_index, prior_delay_ratio, scheduled_flight_duration_min")
print("Phase 4 Feature Engineering Completed Successfully!")
