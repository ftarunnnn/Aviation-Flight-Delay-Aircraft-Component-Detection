# ✈️ Aviation — Flight Delay & Aircraft Component Detection

An end-to-end Machine Learning and Computer Vision platform implementing a **10-Phase Architecture** for predicting flight departure delays and detecting structural aircraft component defects.

---

## 📌 10-Phase System Architecture Overview

```
✈️ AVIATION SYSTEM PIPELINE

[Pipeline A: Flight Delay Prediction (ML)]
Flight Data -> Preprocessing -> EDA -> Feature Engineering -> XGBoost/RandomForest -> Delay & Risk Prediction

[Pipeline B: Aircraft Defect Detection (DL)]
Inspection Images -> Augmentation/Preprocessing -> PyTorch CNN / YOLO -> Bounding Box Localization & Classification

[Phase 10: Unified Interactive Web Dashboard & FastAPI Backend]
+---------------------------------------------------------------------------------+
|  1. Executive Architecture Hub (Interactive 10-Phase Timeline)                  |
|  2. Flight Departure Delay Risk Predictor (ML Inference & Feature Impact)       |
|  3. Aircraft Component Defect Detector (HTML5 Canvas Bounding Box Visualizer)   |
|  4. Operational Analytics & EDA Studio (Chart.js Interactive Visuals)           |
|  5. Model Performance & Evaluation Matrix (MAE, RMSE, IoU, mAP@0.5)             |
|  6. FastAPI REST API Sandbox Console                                            |
+---------------------------------------------------------------------------------+
```

---

## 🛠️ Phase-by-Phase Process & Deliverables Breakdown

| Phase | Process | Description | Core Artifacts |
| :---: | :--- | :--- | :--- |
| **Phase 1** | **Data Collection** | Collect & generate raw flight operational data & inspection panel metadata | `src/phase1_data_collection.py`<br>`data/raw/flights_raw.csv`<br>`data/raw/inspection_images/` |
| **Phase 2** | **Data Preprocessing** | Clean data, handle missing values, encode categoricals & scale image labels | `src/phase2_data_preprocessing.py`<br>`data/processed/flights_clean.csv`<br>`data/processed/inspection_annotations_clean.json` |
| **Phase 3** | **EDA** | Analyze delay factors by weather, congestion, airline & defect frequencies | `src/phase3_eda.py`<br>`reports/eda_report.md`<br>`assets/eda_plots/` |
| **Phase 4** | **Feature Engineering** | Create temporal features, weather severity index & congestion risk scores | `src/phase4_feature_engineering.py`<br>`data/processed/flights_engineered.csv` |
| **Phase 5** | **ML Model Training** | Train Random Forest & XGBoost regressors and classifiers | `src/phase5_ml_training.py`<br>`models/flight_delay_rf.pkl`<br>`models/flight_delay_xgboost_reg.json` |
| **Phase 6** | **ML Evaluation** | Evaluate models using MAE, RMSE, Accuracy, Precision, Recall, F1 & Confusion Matrix | `src/phase6_ml_evaluation.py`<br>`reports/ml_evaluation_metrics.json`<br>`assets/ml_eval/` |
| **Phase 7** | **DL Image Processing** | Image augmentation (flips, contrast) & YOLO bounding box format normalization | `src/phase7_dl_image_processing.py`<br>`data/processed/inspection_augmented/` |
| **Phase 8** | **DL Model Training** | Train PyTorch CNN Defect Classifier & build YOLO detector config | `src/phase8_dl_training.py`<br>`models/defect_classifier_cnn.pt`<br>`models/yolo_detector_config.json` |
| **Phase 9** | **DL Evaluation** | Evaluate defect localization & classification using Mean IoU & mAP@0.5 | `src/phase9_dl_evaluation.py`<br>`reports/dl_evaluation_metrics.json`<br>`assets/dl_eval/` |
| **Phase 10** | **Deployment & Output** | Deploy FastAPI REST backend server and interactive glassmorphism Web Dashboard | `src/api/app.py`<br>`index.html`<br>`styles.css`<br>`app.js` |

---

## 📊 Summary Evaluation Performance Metrics

### ✈️ Flight Delay Prediction (ML - XGBoost / Random Forest)
- **Mean Absolute Error (MAE)**: `9.35 Minutes`
- **Root Mean Squared Error (RMSE)**: `11.94 Minutes`
- **R² Score**: `0.7377`
- **Delay Risk Classification Accuracy**: `80.00%`

### 🔧 Aircraft Component Defect Localization (DL - PyTorch CNN & YOLO)
- **Mean Intersection over Union (Mean IoU)**: `0.4890`
- **mAP@0.5 (mean Average Precision)**: `0.6327`
- **Localization Precision**: `0.5947`
- **Localization Recall**: `0.5757`

---

## ⚡ Quickstart Guide & Local Execution

### 1. Execute Data & ML/DL Pipelines (Phases 1-9)
```bash
# Run Phase 1: Data Collection
python src/phase1_data_collection.py

# Run Phase 2: Preprocessing
python src/phase2_data_preprocessing.py

# Run Phase 3: EDA
python src/phase3_eda.py

# Run Phase 4: Feature Engineering
python src/phase4_feature_engineering.py

# Run Phase 5: ML Training
python src/phase5_ml_training.py

# Run Phase 6: ML Evaluation
python src/phase6_ml_evaluation.py

# Run Phase 7: DL Processing
python src/phase7_dl_image_processing.py

# Run Phase 8: DL Training
python src/phase8_dl_training.py

# Run Phase 9: DL Evaluation
python src/phase9_dl_evaluation.py
```

### 2. Launch FastAPI Backend & Interactive Dashboard (Phase 10)
```bash
# Start FastAPI Server
python src/api/app.py

# Launch Dashboard UI
python -m http.server 8080
```
Open `http://localhost:8080` in your web browser.

---

## 📜 License
MIT License - Created for Aviation Analytics & Aircraft Inspection Automation.