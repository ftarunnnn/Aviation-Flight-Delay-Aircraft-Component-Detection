"""
Phase 6: ML Evaluation Script
Aviation - Flight Delay & Aircraft Component Detection
Evaluates trained XGBoost & Random Forest models computing MAE, RMSE, Accuracy, Precision, Recall, F1-score & feature importances.
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
PLOTS_DIR = os.path.join(BASE_DIR, "assets", "ml_eval")

os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

print("Starting Phase 6: ML Evaluation...")

# Load Data and Feature Metadata
df = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, "flights_engineered.csv"))
with open(os.path.join(MODELS_DIR, "feature_columns.json"), 'r', encoding='utf-8') as f:
    feat_meta = json.load(f)

FEATURE_COLS = feat_meta["features"]
X = df[FEATURE_COLS]
y_reg = df['actual_departure_delay_min']
y_cls = df['delay_risk_class']

_, X_test, _, y_reg_test, _, y_cls_test = train_test_split(
    X, y_reg, y_cls, test_size=0.2, random_state=42, stratify=y_cls
)

# Load Models
rf_path = os.path.join(MODELS_DIR, "flight_delay_rf.pkl")
with open(rf_path, 'rb') as f:
    rf_models = pickle.load(f)
rf_reg = rf_models["regressor"]
rf_cls = rf_models["classifier"]

# Predict with Random Forest
rf_reg_pred = rf_reg.predict(X_test)
rf_cls_pred = rf_cls.predict(X_test)

# Predict with XGBoost if available
if HAS_XGBOOST and os.path.exists(os.path.join(MODELS_DIR, "flight_delay_xgboost_reg.json")):
    xgb_reg = xgb.XGBRegressor()
    xgb_reg.load_model(os.path.join(MODELS_DIR, "flight_delay_xgboost_reg.json"))
    xgb_cls = xgb.XGBClassifier()
    xgb_cls.load_model(os.path.join(MODELS_DIR, "flight_delay_xgboost_cls.json"))
    
    xgb_reg_pred = xgb_reg.predict(X_test)
    xgb_cls_pred = xgb_cls.predict(X_test)
    best_reg_pred = xgb_reg_pred
    best_cls_pred = xgb_cls_pred
    model_name = "XGBoost"
else:
    best_reg_pred = rf_reg_pred
    best_cls_pred = rf_cls_pred
    model_name = "RandomForest"

# Compute Regression Metrics
mae = float(mean_absolute_error(y_reg_test, best_reg_pred))
rmse = float(np.sqrt(mean_squared_error(y_reg_test, best_reg_pred)))
r2 = float(r2_score(y_reg_test, best_reg_pred))

# Compute Classification Metrics
acc = float(accuracy_score(y_cls_test, best_cls_pred))
prec = float(precision_score(y_cls_test, best_cls_pred, average='weighted'))
rec = float(recall_score(y_cls_test, best_cls_pred, average='weighted'))
f1 = float(f1_score(y_cls_test, best_cls_pred, average='weighted'))
conf_mat = confusion_matrix(y_cls_test, best_cls_pred).tolist()

metrics = {
    "model_type": model_name,
    "regression_metrics": {
        "MAE_minutes": round(mae, 2),
        "RMSE_minutes": round(rmse, 2),
        "R2_score": round(r2, 4)
    },
    "classification_metrics": {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "confusion_matrix": conf_mat
    }
}

metrics_path = os.path.join(REPORTS_DIR, "ml_evaluation_metrics.json")
with open(metrics_path, 'w', encoding='utf-8') as f:
    json.dump(metrics, f, indent=2)

print(f"ML Evaluation metrics saved to: {metrics_path}")
print(f"Regression -> MAE: {mae:.2f} mins | RMSE: {rmse:.2f} mins | R2: {r2:.4f}")
print(f"Classification -> Accuracy: {acc*100:.2f}% | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}")

# Plot 1: Confusion Matrix
plt.figure(figsize=(7, 6))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Low Risk', 'Moderate Risk', 'High Risk'],
            yticklabels=['Low Risk', 'Moderate Risk', 'High Risk'])
plt.title(f'Flight Delay Risk Classification Confusion Matrix ({model_name})', fontsize=12, fontweight='bold', pad=12)
plt.xlabel('Predicted Delay Risk Category', fontsize=11)
plt.ylabel('Actual Delay Risk Category', fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "confusion_matrix.png"), dpi=300)
plt.close()

# Plot 2: Feature Importance
plt.figure(figsize=(10, 6))
importances = rf_cls.feature_importances_
indices = np.argsort(importances)[::-1]
sorted_cols = [FEATURE_COLS[i] for i in indices]
sorted_importances = importances[indices]

sns.barplot(x=sorted_importances[:10], y=sorted_cols[:10], hue=sorted_cols[:10], legend=False, palette='mako')
plt.title('Top 10 Feature Importances for Flight Delay Prediction', fontsize=13, fontweight='bold', pad=12)
plt.xlabel('Gini Importance Score', fontsize=11)
plt.ylabel('Feature Name', fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "feature_importance.png"), dpi=300)
plt.close()

# Plot 3: Actual vs Predicted Delay Minutes
plt.figure(figsize=(8, 6))
plt.scatter(y_reg_test, best_reg_pred, alpha=0.6, color='#2563eb', edgecolors='k', linewidth=0.5)
plt.plot([0, max(y_reg_test)], [0, max(y_reg_test)], 'r--', label='Ideal Fit')
plt.title(f'Actual vs Predicted Flight Departure Delay ({model_name})', fontsize=12, fontweight='bold')
plt.xlabel('Actual Delay (Minutes)', fontsize=11)
plt.ylabel('Predicted Delay (Minutes)', fontsize=11)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "actual_vs_predicted.png"), dpi=300)
plt.close()

print(f"ML evaluation plots saved to: {PLOTS_DIR}")
print("Phase 6 ML Evaluation Completed Successfully!")
