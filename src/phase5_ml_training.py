"""
Phase 5: ML Model Training Script
Aviation - Flight Delay & Aircraft Component Detection
Trains Random Forest and XGBoost Classifier/Regressor models for flight delay prediction and risk scoring.
"""

import os
import json
import pickle
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(MODELS_DIR, exist_ok=True)

print("Starting Phase 5: ML Model Training...")

# Load Engineered Dataset
df = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, "flights_engineered.csv"))

# Select Features for ML Training
FEATURE_COLS = [
    'airline_code', 'weather_code', 'temp_norm', 'wind_norm', 'traffic_norm',
    'prior_leg_delay_min', 'aircraft_age_years', 'maintenance_overdue_flag',
    'dep_hour', 'dep_day_of_week', 'is_weekend', 'is_peak_hours',
    'weather_severity_score', 'congestion_risk_score', 'aircraft_wear_index',
    'prior_delay_ratio'
]

X = df[FEATURE_COLS]
y_reg = df['actual_departure_delay_min']
y_cls = df['delay_risk_class']

X_train, X_test, y_reg_train, y_reg_test, y_cls_train, y_cls_test = train_test_split(
    X, y_reg, y_cls, test_size=0.2, random_state=42, stratify=y_cls
)

# 1. Random Forest Training
print("Training Random Forest Regressor & Classifier...")
rf_reg = RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42)
rf_reg.fit(X_train, y_reg_train)

rf_cls = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42)
rf_cls.fit(X_train, y_cls_train)

# Save Random Forest Models
rf_model_path = os.path.join(MODELS_DIR, "flight_delay_rf.pkl")
with open(rf_model_path, 'wb') as f:
    pickle.dump({"regressor": rf_reg, "classifier": rf_cls}, f)
print(f"Random Forest model persisted at: {rf_model_path}")

# 2. XGBoost Training (or Gradient Boosting Fallback)
xgb_reg_pred, xgb_cls_pred = None, None
if HAS_XGBOOST:
    print("Training XGBoost Regressor & Classifier...")
    xgb_reg = xgb.XGBRegressor(n_estimators=120, max_depth=6, learning_rate=0.08, random_state=42)
    xgb_reg.fit(X_train, y_reg_train)
    xgb_reg_path = os.path.join(MODELS_DIR, "flight_delay_xgboost_reg.json")
    xgb_reg.save_model(xgb_reg_path)
    
    xgb_cls = xgb.XGBClassifier(n_estimators=120, max_depth=6, learning_rate=0.08, random_state=42)
    xgb_cls.fit(X_train, y_cls_train)
    xgb_cls_path = os.path.join(MODELS_DIR, "flight_delay_xgboost_cls.json")
    xgb_cls.save_model(xgb_cls_path)
    print(f"XGBoost models persisted at: {MODELS_DIR}")
else:
    from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
    print("XGBoost package not found, using Gradient Boosting Classifier/Regressor...")
    gb_reg = GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)
    gb_reg.fit(X_train, y_reg_train)
    gb_cls = GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)
    gb_cls.fit(X_train, y_cls_train)
    
    gb_model_path = os.path.join(MODELS_DIR, "flight_delay_xgboost.pkl")
    with open(gb_model_path, 'wb') as f:
        pickle.dump({"regressor": gb_reg, "classifier": gb_cls}, f)

# Save Feature Metadata
feature_meta_path = os.path.join(MODELS_DIR, "feature_columns.json")
with open(feature_meta_path, 'w') as f:
    json.dump({
        "features": FEATURE_COLS,
        "feature_count": len(FEATURE_COLS),
        "train_samples": len(X_train),
        "test_samples": len(X_test)
    }, f, indent=2)

print(f"Saved feature metadata at: {feature_meta_path}")
print("Phase 5 ML Model Training Completed Successfully!")
