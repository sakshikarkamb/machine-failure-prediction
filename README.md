# Machine Failure Prediction — AI-Based Predictive Maintenance System

A complete end-to-end Machine Learning and Predictive Maintenance web application designed to anticipate industrial machine failures before they occur. Powered by a tuned Random Forest pipeline, a robust FastAPI backend service, and an interactive Streamlit dashboard.

---

## 📌 Problem Statement

In modern manufacturing, unexpected machinery breakdowns cause costly unplanned downtime, production bottlenecks, and hazardous working conditions. Traditional maintenance strategies either:
- **Run-to-failure (Reactive):** Wait until breakdown, leading to expensive repairs and lost productivity.
- **Scheduled Maintenance (Preventive):** Service equipment at rigid intervals regardless of condition, causing unnecessary part replacements and labor expenses.

**Predictive Maintenance (PdM)** uses real-time operating telemetry (temperatures, speed, torque, tool wear) to predict failures dynamically, allowing maintenance teams to intervene proactively.

---

## 🎯 Project Objective

- Build an industrial predictive maintenance system capable of accurately classifying whether a machine is at risk of failure (`Machine failure = 1`) or operating nominally (`Machine failure = 0`).
- Deploy the untouched trained model through a high-performance **FastAPI** REST backend.
- Provide operators with an intuitive **Streamlit** dashboard showing real-time risk probabilities, parameter summaries, model metrics, feature importances, and historical logs.

---

## 📊 Dataset & Features

The model is trained on the standard **AI4I 2020 Predictive Maintenance Dataset** (`Data/raw/ai4i2020.csv`), reflecting real milling machine operations:

| Feature Name | Type | Description |
| :--- | :--- | :--- |
| **Type** | Categorical | Machine product variant: `L` (Low - 50%), `M` (Medium - 30%), `H` (High - 20%) |
| **Air temperature [K]** | Continuous | Ambient operational temperature in Kelvin |
| **Process temperature [K]** | Continuous | Internal process heat in Kelvin |
| **Rotational speed [rpm]** | Continuous | Spindle rotation velocity in revolutions per minute |
| **Torque [Nm]** | Continuous | Torque delivered by the spindle motor in Newton-meters |
| **Tool wear [min]** | Continuous | Cumulative wear time of the cutting tool in minutes |
| **Machine failure** | Binary Target | `0` = Nominal Operation, `1` = Machine Failure Occurred |

---

## 🔬 Machine Learning Workflow

1. **Exploratory Data Analysis (EDA):** Analyzed multi-modal sensor distributions, identified high class imbalance (~3.4% failure rate), and uncovered strong correlations between rotational speed/torque power products and failure modes.
2. **Preprocessing Pipeline (`ColumnTransformer`):**
   - **One-Hot Encoding** (`handle_unknown='ignore'`) for categorical feature `Type`.
   - **StandardScaler** for numerical sensors (`Air temperature`, `Process temperature`, `Rotational speed`, `Torque`, `Tool wear`).
3. **Model Exploration & Comparison:** Evaluated baseline algorithms including Logistic Regression, Decision Trees, and Random Forest ensembles.
4. **Hyperparameter Tuning:** Tuned Random Forest with `class_weight='balanced'`, `n_estimators=200`, `max_depth=20`, and `min_samples_leaf=2` to optimize sensitivity on minority failure events.
5. **Serialization:** Saved full inference pipeline as `models/machine_failure_random_forest.pkl`.

---

## 📈 Evaluation Metrics (Untouched Test Evaluation)

Evaluated on the 2,000-sample test set:

| Metric | Score |
| :--- | :--- |
| **Accuracy** | **97.40%** |
| **Precision** | **58.89%** |
| **Recall** | **77.94%** |
| **F1 Score** | **67.09%** |
| **ROC-AUC** | **96.26%** |

### Confusion Matrix
```
                  Predicted Normal (0)   Predicted Failure (1)
Actual Normal (0)         1895                    37
Actual Failure (1)          15                    53
```
- **True Negatives (TN):** `1,895`
- **False Positives (FP):** `37` (False alarms)
- **False Negatives (FN):** `15` (Missed failures)
- **True Positives (TP):** `53` (Caught failures)

---

## 🌟 Feature Importance

Extracted directly from the trained Random Forest classifier:
1. **Torque [Nm]:** ~31.61%
2. **Rotational speed [rpm]:** ~29.55%
3. **Tool wear [min]:** ~20.67%
4. **Air temperature [K]:** ~9.82%
5. **Process temperature [K]:** ~6.67%
6. **Machine Type (L / M / H):** ~1.68%

---

## 🏗️ System Architecture

```text
┌─────────────────────────────────┐
│       Streamlit Frontend        │  <--- Web UI (Port 8501)
│        (frontend/app.py)        │
└────────────────┬────────────────┘
                 │ HTTP Requests (JSON)
                 ▼
┌─────────────────────────────────┐
│         FastAPI Backend         │  <--- REST API (Port 8000)
│        (backend/main.py)        │
└────────────────┬────────────────┘
                 │ Loads & Queries
                 ▼
┌─────────────────────────────────┐
│     Saved RF Model Pipeline     │  <--- models/machine_failure_random_forest.pkl
│  (ColumnTransformer + Classifier│
└─────────────────────────────────┘
```

---

## 🔌 Backend API Documentation

### Endpoints
* **`GET /`** — Health and welcome message.
* **`GET /health`** — Service status and model availability confirmation.
* **`POST /predict`** — Generates binary prediction and failure risk percentage.
* **`GET /model-info`** — Returns pipeline steps and extracted feature importances.
* **`GET /docs`** — Interactive Swagger API documentation.

### Sample Prediction Request
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "type": "M",
       "air_temperature": 300.0,
       "process_temperature": 310.0,
       "rotational_speed": 1500.0,
       "torque": 40.0,
       "tool_wear": 100.0
     }'
```

### Sample Prediction Response
```json
{
  "prediction": 0,
  "result": "No Machine Failure",
  "failure_probability": 0.0
}
```

---

## 🚀 How to Run the Application

### 1. Prerequisites
Python 3.10+ (tested with Python 3.13 / 3.14).

### 2. Install Dependencies
```bash
# Backend dependencies
pip install -r backend/requirements.txt

# Frontend dependencies
pip install -r frontend/requirements.txt
```

### 3. Run Automated Tests
```bash
python backend/test_api.py
```

### 4. Start the FastAPI Backend
```bash
uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```
Swagger UI is available at: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

### 5. Start the Streamlit Frontend
In a new terminal window:
```bash
streamlit run frontend/app.py
```
Dashboard will open at: [http://localhost:8501](http://localhost:8501)

---

## 📸 Screenshots

*(Screenshots can be placed in `screenshots/` directory)*
- **Live Prediction Interface:** Input sliders, quick presets, real-time risk gauge.
- **Analytics & Metrics:** Confusion matrix, accuracy, recall, and feature importance rankings.
- **Prediction History:** Session history with CSV export capabilities.

---

## 🔮 Future Scope

- **Multi-failure mode classification:** Predict specific failure modes (Tool Wear Failure, Heat Dissipation Failure, Power Failure, Overstrain Failure).
- **IoT Streaming Ingestion:** Connect Kafka or MQTT broker for real-time live sensor streaming.
- **Historical Database:** Persist telemetry and predictions into PostgreSQL / TimescaleDB for continuous drift monitoring.
