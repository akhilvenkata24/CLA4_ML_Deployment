# ML Model Deployment Strategies and Model Serving using FastAPI

**Assignment**: CLA 4  
**Dataset**: [Kaggle Titanic: Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic/data)  
**Trained Model**: Complete Preprocessing + Random Forest Pipeline (`model.pkl`)  
**Actual Test Accuracy**: **82.12%**  

---

## 📋 Project Overview

This repository demonstrates real-world machine learning deployment paradigms and production model serving using the official Kaggle Titanic dataset. The workflow trains a scikit-learn classification pipeline once, serializes it to `model.pkl`, and reuses that exact model across all downstream inference strategies:

1. **Model Training & Preprocessing**: Automated handling of missing values, feature scaling, and one-hot encoding without data leakage.
2. **Online Inference (`online_inference.py`)**: Real-time single-record prediction for a single unseen passenger.
3. **Batch Inference (`batch_inference.py`)**: Vectorized scoring of 20 unseen passenger records, exporting predictions to `batch_predictions.csv`.
4. **Streaming Inference (`streaming_inference.py`)**: Event-driven sequential processing of 10 unseen passenger records with a 1-second delay between records.
5. **FastAPI Model Serving (`app.py` & `client.py`)**: REST API with `GET /health` and `POST /predict`, automatic Pydantic request validation, and interactive Swagger UI documentation at `/docs`.
6. **Academic Observations**: Comparative analysis of inference strategies and architectural justification for FastAPI and model reuse.

---

## 📁 Repository Structure

```text
CLA4_ML_Deployment/
├── data/
│   ├── train.csv                   # Official Kaggle Titanic training dataset (891 rows)
│   ├── test.csv                    # Kaggle test set
│   ├── gender_submission.csv       # Kaggle benchmark submission
│   └── unseen_test_records.csv     # 179 genuine unseen test records isolated for evaluation
├── screenshots/
│   ├── 01_model_training.png       # Execution of train_model.py
│   ├── 02_online_inference.png     # Execution of online_inference.py
│   ├── 03_batch_inference.png      # Execution of batch_inference.py
│   ├── 04_streaming_inference.png  # Execution of streaming_inference.py
│   ├── 05_fastapi_swagger_docs.png # Swagger UI at http://127.0.0.1:8000/docs
│   ├── 06_fastapi_endpoints_expanded.png # Expanded /health and /predict in Swagger
│   ├── 07_fastapi_health_endpoint.png   # Browser view of /health endpoint
│   └── 08_fastapi_client_test.png  # Automated execution of client.py
├── app.py                          # FastAPI model serving backend
├── client.py                       # Automated test client for FastAPI
├── train_model.py                  # Model training and artifact generation script
├── online_inference.py             # Single-record online inference
├── batch_inference.py              # 20-record batch inference
├── streaming_inference.py          # 10-event simulated streaming inference
├── generate_screenshots.py         # Automated screenshot capture script
├── generate_report.py              # Professional academic PDF report builder
├── batch_predictions.csv           # 20 generated batch predictions
├── model.pkl                       # Serialized fitted pipeline (preprocessing + classifier)
├── requirements.txt                # Python package dependencies
├── ML_Deployment_Observations_Report.pdf # Comprehensive academic PDF report
├── report.pdf                      # Convenience copy of the assignment PDF report
└── README.md                       # Complete documentation
```

---

## ⚙️ Installation & Setup

### 1. Prerequisites
- Python 3.10+ (Tested on Python 3.12.7)
- Virtual environment (recommended)

### 2. Clone the Repository
```bash
git clone <repository_url>
cd CLA4_ML_Deployment
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 Execution Guide

### Task 1: Train the Classification Model
Trains the preprocessing pipeline and Random Forest model once, saves `model.pkl`, and creates `data/unseen_test_records.csv`:
```bash
python train_model.py
```
**Actual Output:**
- Raw records loaded: 891 rows, 12 columns
- Training records: 712 (79.9%)
- Testing records: 179 (20.1%)
- **Actual Test Accuracy**: **82.12%**
- Model artifact saved to: `model.pkl`

---

### Task 2: Online Inference (Single Record)
Loads `model.pkl` and predicts survival for a single genuine unseen passenger in real time:
```bash
python online_inference.py
```
**Actual Output:**
- Passenger: 3rd Class Male, 24 years old, Fare $24.15
- Prediction: **Did Not Survive (0)** (Survival Probability: 11.28%)
- Ground Truth: Did Not Survive (0) — **CORRECT**

---

### Task 3: Batch Inference (20 Records)
Scores 20 genuine unseen passenger records simultaneously and saves results to `batch_predictions.csv`:
```bash
python batch_inference.py
```
**Actual Output:**
- Total Processed: 20 records
- Batch Accuracy: **85.0%** (17 / 20 correct)
- Output File: `batch_predictions.csv`

---

### Task 4: Streaming Inference (10 Events)
Simulates a real-time event pipeline by scoring 10 unseen passenger records with a 1.0-second delay between records:
```bash
python streaming_inference.py
```
**Actual Output:**
- Total Events Streamed: 10
- Stream Accuracy: **70.0%** (7 / 10 correct)
- Delay: 1.0 second per event (~10 seconds total elapsed)

---

### Task 5: FastAPI Model Serving

#### 1. Start the API Server
```bash
uvicorn app:app --host 127.0.0.1 --port 8000
```
- Interactive Swagger UI: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- Alternative ReDoc: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)
- Health Check: [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)

#### 2. Run the Automated Client Suite
In a new terminal window:
```bash
python client.py
```
**Test Results:**
- `GET /health` ➔ **200 OK** (Model loaded: `RandomForestClassifier`)
- `POST /predict` (1st Class Female) ➔ **200 OK** (`Survived`, Prob: 99.13%)
- `POST /predict` (3rd Class Male) ➔ **200 OK** (`Did Not Survive`, Prob: 11.28%)
- `POST /predict` (Invalid Payload) ➔ **422 Unprocessable Entity** (Pydantic schema validation caught errors)

---

## 💡 Technical Observations & Answers to Assignment Questions

### 1. Online Inference vs. Batch Inference
* **Online Inference**:
  * **Latency**: Ultra-low (milliseconds).
  * **Processing**: Single record or micro-batch on-demand.
  * **Architecture**: Synchronous REST APIs, WebSockets, or gRPC.
  * **Use Cases**: User-facing applications (e.g., e-commerce fraud detection, live recommendation carousels).
* **Batch Inference**:
  * **Latency**: High (minutes to hours).
  * **Processing**: Large static datasets grouped into chunks (thousands/millions of records).
  * **Architecture**: Scheduled background jobs (e.g., Apache Airflow, cron).
  * **Use Cases**: Nightly billing churn scoring, risk assessment reports, database pre-computation.

### 2. Streaming Inference vs. Batch Inference
* **Streaming Inference**:
  * **Data**: Continuous, unbounded event feeds (Kafka, AWS Kinesis, RabbitMQ).
  * **Responsiveness**: Near real-time per event as messages arrive.
  * **State**: Often maintains sliding windows or event state; resilient to consumer restarts.
  * **Use Cases**: IoT anomaly detection, telemetry monitoring, real-time alert systems.
* **Batch Inference**:
  * **Data**: Static, bounded historical datasets stored on disk/database.
  * **Responsiveness**: Periodic; delayed until the batch interval fires.
  * **Throughput**: Maximum computational throughput via vectorized array operations.

### 3. Why FastAPI is Used for Model Serving
1. **Asynchronous Performance**: Built on ASGI (Starlette) and Uvicorn, offering throughput comparable to Node.js and Go.
2. **Built-in Type Validation**: Powered by Pydantic; guarantees incoming request types, bounds, and required attributes are strictly verified before touching ML logic.
3. **Automatic Documentation**: Automatically generates interactive OpenAPI Swagger UI (`/docs`) and ReDoc without writing manual schema files.
4. **Python-Native ML Interoperability**: Seamlessly hosts `scikit-learn`, `joblib`, `pandas`, and `numpy` objects directly in application process memory.

### 4. Why the Same Trained Model Must Be Reused for Every Inference Strategy
1. **Elimination of Training-Serving Skew**: Different models or retraining runs introduce distribution drift, varying random seeds, and incompatible preprocessing states (e.g., differing scaler statistics or one-hot category mappings). Reusing `model.pkl` guarantees mathematically identical predictions.
2. **Single Source of Truth**: Simplifies enterprise MLOps governance, versioning, rollback, and regulatory auditing.
3. **Operational & Computational Efficiency**: Avoids redundant training overhead and maximizes GPU/CPU resource utilization.

---

## 📊 Summary of Actual Execution Metrics

| Task | Component | Metric / Result | Status |
| :--- | :--- | :--- | :---: |
| **Task 1** | Model Training | Test Accuracy: **82.12%** | ✅ Verified |
| **Task 2** | Online Inference | Prediction: Did Not Survive (11.28% prob) | ✅ Verified |
| **Task 3** | Batch Inference | 20 records; Accuracy: **85.0%** | ✅ Verified |
| **Task 4** | Streaming Inference | 10 events (1s delay); Accuracy: **70.0%** | ✅ Verified |
| **Task 5** | FastAPI /health | HTTP 200 OK; Model loaded | ✅ Verified |
| **Task 5** | FastAPI /predict (Valid) | HTTP 200 OK; Correct classifications | ✅ Verified |
| **Task 5** | FastAPI /predict (Invalid) | HTTP 422 Unprocessable Entity | ✅ Verified |
| **Task 6** | Academic Report | Generated: `ML_Deployment_Observations_Report.pdf` | ✅ Verified |

---

## 📄 License & Attribution
This project is developed for educational purposes as part of the MLOps CLA 4 college coursework. Dataset provided by Kaggle Titanic Competition under its respective educational terms.
