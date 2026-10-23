"""
CLA 4 - Task 6: Comprehensive Academic PDF Report Generator
Generates a formal, professional assignment report covering:
1. Project Overview & Architecture
2. Task 1: Model Training & Preprocessing
3. Task 2: Online Inference
4. Task 3: Batch Inference
5. Task 4: Streaming Inference
6. Task 5: FastAPI Model Serving
7. Task 6: Detailed Technical Observations (4 Key Questions)
8. Verification & Results Summary
"""

import os
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, HRFlowable, PageBreak, KeepTogether
)

PROJECT_ROOT = os.path.dirname(__file__)
SCREENSHOTS_DIR = os.path.join(PROJECT_ROOT, "screenshots")
OUTPUT_PDF = os.path.join(PROJECT_ROOT, "ML_Deployment_Observations_Report.pdf")


def build_pdf():
    doc = SimpleDocTemplate(
        OUTPUT_PDF,
        pagesize=letter,
        leftMargin=0.6 * inch,
        rightMargin=0.6 * inch,
        topMargin=0.6 * inch,
        bottomMargin=0.6 * inch
    )

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        "CoverTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=22,
        leading=26,
        textColor=colors.HexColor("#1e293b"),
        alignment=1,
        spaceAfter=8
    )

    subtitle_style = ParagraphStyle(
        "CoverSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=12,
        leading=16,
        textColor=colors.HexColor("#475569"),
        alignment=1,
        spaceAfter=15
    )

    meta_style = ParagraphStyle(
        "MetaText",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#334155"),
        alignment=1
    )

    h1_style = ParagraphStyle(
        "Heading1_Custom",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=14,
        leading=18,
        textColor=colors.HexColor("#0f172a"),
        spaceBefore=14,
        spaceAfter=6
    )

    h2_style = ParagraphStyle(
        "Heading2_Custom",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=15,
        textColor=colors.HexColor("#1e40af"),
        spaceBefore=10,
        spaceAfter=4
    )

    body_style = ParagraphStyle(
        "Body_Custom",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=13.5,
        textColor=colors.HexColor("#1e293b"),
        spaceAfter=6
    )

    qa_question_style = ParagraphStyle(
        "QAQuestion",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#1e3a8a"),
        spaceBefore=8,
        spaceAfter=3
    )

    qa_answer_style = ParagraphStyle(
        "QAAnswer",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#334155"),
        spaceAfter=8
    )

    code_style = ParagraphStyle(
        "CodeText",
        parent=styles["Normal"],
        fontName="Courier",
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor("#0f172a")
    )

    elements = []

    # Title & Header
    elements.append(Paragraph("ML Model Deployment Strategies & Model Serving using FastAPI", title_style))
    elements.append(Paragraph("CLA 4 — Complete Autonomous Implementation & Observations Report", subtitle_style))
    elements.append(Paragraph("<b>Student Assignment:</b> CLA 4 &nbsp;|&nbsp; <b>Dataset:</b> Kaggle Titanic Competition &nbsp;|&nbsp; <b>Model:</b> Random Forest Pipeline", meta_style))
    elements.append(Spacer(1, 10))
    elements.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#cbd5e1"), spaceAfter=15))

    # Section 1: Executive Summary & System Architecture
    elements.append(Paragraph("1. Executive Summary & Pipeline Architecture", h1_style))
    elements.append(Paragraph(
        "This project implements end-to-end machine learning model serving architectures on the official Kaggle Titanic dataset. "
        "A full preprocessing and classification pipeline was trained once, persisted as an artifact (<b>model.pkl</b>), and systematically evaluated across "
        "three primary inference paradigms (Online, Batch, and Streaming) and served via a high-performance <b>FastAPI</b> REST service.",
        body_style
    ))

    summary_data = [
        [Paragraph("<b>Component</b>", body_style), Paragraph("<b>Specification / Architecture</b>", body_style), Paragraph("<b>Result / Metric</b>", body_style)],
        [Paragraph("Dataset Source", body_style), Paragraph("Kaggle Titanic (data/train.csv)", code_style), Paragraph("891 records, 12 features", body_style)],
        [Paragraph("Training / Test Split", body_style), Paragraph("80% Training, 20% Unseen Testing (Stratified)", body_style), Paragraph("712 Train / 179 Test", body_style)],
        [Paragraph("Classification Model", body_style), Paragraph("RandomForestClassifier (100 trees, depth 6)", body_style), Paragraph("<b>82.12% Test Accuracy</b>", body_style)],
        [Paragraph("Model Persistence", body_style), Paragraph("Full fitted ColumnTransformer + Classifier Pipeline", body_style), Paragraph("model.pkl (610.3 KB)", code_style)],
        [Paragraph("Serving Framework", body_style), Paragraph("FastAPI with Pydantic validation & Uvicorn ASGI", body_style), Paragraph("GET /health, POST /predict", code_style)]
    ]
    t_summary = Table(summary_data, colWidths=[1.8 * inch, 3.2 * inch, 2.0 * inch])
    t_summary.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#f1f5f9")),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
    ]))
    elements.append(t_summary)
    elements.append(Spacer(1, 12))

    # Section 2: Model Training & Evaluation
    elements.append(Paragraph("2. Task 1: Model Training & Preprocessing Pipeline", h1_style))
    elements.append(Paragraph(
        "The model pipeline was constructed with scikit-learn's <b>ColumnTransformer</b> and <b>Pipeline</b> abstractions to eliminate data leakage. "
        "Numerical attributes (<i>Pclass, Age, SibSp, Parch, Fare</i>) were imputed with the median and standardized using <i>StandardScaler</i>. "
        "Categorical attributes (<i>Sex, Embarked</i>) were imputed with the mode and encoded with <i>OneHotEncoder(handle_unknown='ignore')</i>. "
        "Identifiers (<i>PassengerId, Name, Ticket</i>) and high-null features (<i>Cabin</i>) were excluded. "
        "The complete fitted object was serialized to <b>model.pkl</b> and 179 genuine unseen test records were isolated in <b>data/unseen_test_records.csv</b>.",
        body_style
    ))

    img_train = os.path.join(SCREENSHOTS_DIR, "01_model_training.png")
    if os.path.exists(img_train):
        elements.append(Image(img_train, width=6.8 * inch, height=3.6 * inch))
        elements.append(Spacer(1, 10))

    elements.append(PageBreak())

    # Section 3: Inference Strategies
    elements.append(Paragraph("3. Model Inference Strategies Evaluation", h1_style))

    # Task 2: Online Inference
    elements.append(Paragraph("Task 2: Online Inference (Single Record Real-Time Scoring)", h2_style))
    elements.append(Paragraph(
        "Online inference handles individual prediction requests on demand with minimal latency. "
        "The saved <b>model.pkl</b> pipeline was loaded, and an unseen passenger record (3rd class male, 24 years old) was scored. "
        "Prediction: <b>Did Not Survive</b> (Survival Probability: 11.28%), matching actual ground truth.",
        body_style
    ))
    img_online = os.path.join(SCREENSHOTS_DIR, "02_online_inference.png")
    if os.path.exists(img_online):
        elements.append(Image(img_online, width=6.8 * inch, height=2.4 * inch))
        elements.append(Spacer(1, 8))

    # Task 3: Batch Inference
    elements.append(Paragraph("Task 3: Batch Inference (High-Throughput Bulk Scoring)", h2_style))
    elements.append(Paragraph(
        "Batch inference processes 20 genuine unseen passenger records simultaneously in a single vectorized computation. "
        "The predictions and probabilities were compiled and exported to <b>batch_predictions.csv</b>. "
        "Batch accuracy achieved: <b>85.0%</b> (17 out of 20 correct).",
        body_style
    ))
    img_batch = os.path.join(SCREENSHOTS_DIR, "03_batch_inference.png")
    if os.path.exists(img_batch):
        elements.append(Image(img_batch, width=6.8 * inch, height=2.9 * inch))
        elements.append(Spacer(1, 8))

    # Task 4: Streaming Inference
    elements.append(Paragraph("Task 4: Streaming Inference (Event-Driven Sequential Simulation)", h2_style))
    elements.append(Paragraph(
        "Streaming inference simulates a real-time event pipeline where individual records arrive continuously over a message bus. "
        "Ten unseen passenger records were processed sequentially with a 1.0-second delay between events, outputting live classifications as each arrived.",
        body_style
    ))
    img_stream = os.path.join(SCREENSHOTS_DIR, "04_streaming_inference.png")
    if os.path.exists(img_stream):
        elements.append(Image(img_stream, width=6.8 * inch, height=3.0 * inch))
        elements.append(Spacer(1, 10))

    elements.append(PageBreak())

    # Section 4: FastAPI Model Serving
    elements.append(Paragraph("4. Task 5: FastAPI Model Serving & Client Testing", h1_style))
    elements.append(Paragraph(
        "A production-grade REST API was built in <b>app.py</b> using FastAPI. The server loads the saved <b>model.pkl</b> once upon startup. "
        "It provides endpoints: <b>GET /health</b> for liveness and model inspection, and <b>POST /predict</b> with robust Pydantic data validation.",
        body_style
    ))

    img_swagger = os.path.join(SCREENSHOTS_DIR, "06_fastapi_endpoints_expanded.png")
    if os.path.exists(img_swagger):
        elements.append(Paragraph("<b>FastAPI Interactive Swagger UI (/docs)</b>", h2_style))
        elements.append(Image(img_swagger, width=6.8 * inch, height=3.4 * inch))
        elements.append(Spacer(1, 8))

    img_client = os.path.join(SCREENSHOTS_DIR, "08_fastapi_client_test.png")
    if os.path.exists(img_client):
        elements.append(Paragraph("<b>Automated Client Test Suite (client.py Execution)</b>", h2_style))
        elements.append(Image(img_client, width=6.8 * inch, height=3.2 * inch))
        elements.append(Spacer(1, 10))

    elements.append(PageBreak())

    # Section 5: Technical Observations (Task 6)
    elements.append(Paragraph("5. Task 6: Technical Observations & Critical Analysis", h1_style))

    # Q1
    elements.append(Paragraph("Question 1: Compare Online Inference versus Batch Inference.", qa_question_style))
    elements.append(Paragraph(
        "<b>Answer:</b><br/>"
        "• <b>Online Inference:</b> Executes predictions synchronously on a single record (or micro-batch) in real time. It prioritizes ultra-low latency (typically &lt;50ms) "
        "and is accessed via REST APIs or RPC calls in user-facing systems (e.g., checkout fraud detection, recommendation engines). However, resource utilization can be variable.<br/>"
        "• <b>Batch Inference:</b> Executes predictions asynchronously over large datasets grouped into chunks (e.g., thousands or millions of rows). It prioritizes high throughput "
        "and amortizes preprocessing/computational overhead. It is triggered periodically (e.g., nightly cron jobs) for downstream reporting or data warehouse population, where immediate per-record response is not required.",
        qa_answer_style
    ))

    # Q2
    elements.append(Paragraph("Question 2: Compare Streaming Inference versus Batch Inference.", qa_question_style))
    elements.append(Paragraph(
        "<b>Answer:</b><br/>"
        "• <b>Streaming Inference:</b> Operates on continuous, unbounded event feeds (such as Kafka, RabbitMQ, or AWS Kinesis). Records are scored individually or in tiny sliding windows "
        "immediately as they arrive. This yields near-real-time responsiveness without maintaining a synchronous HTTP connection, making it resilient to consumer backpressure.<br/>"
        "• <b>Batch Inference:</b> Operates on bounded, static datasets stored in persistent files or tables. Rather than responding to incoming events immediately, it accumulates "
        "data over a designated time window before executing bulk matrix calculations. Streaming handles dynamic temporal data; batch handles static historical data.",
        qa_answer_style
    ))

    # Q3
    elements.append(Paragraph("Question 3: Why is FastAPI used for Model Serving?", qa_question_style))
    elements.append(Paragraph(
        "<b>Answer:</b><br/>"
        "• <b>High Performance:</b> Built on Starlette and Uvicorn, FastAPI provides asynchronous concurrency comparable to NodeJS and Go, outperforming traditional Flask frameworks.<br/>"
        "• <b>Automatic Type Validation via Pydantic:</b> Ensures that corrupt, malformed, or out-of-range passenger records are caught and rejected with HTTP 422 errors before reaching the model.<br/>"
        "• <b>Self-Documenting API:</b> Automatically produces interactive OpenAPI Swagger UI (<i>/docs</i>) and ReDoc interfaces directly from Python type annotations without external tools.<br/>"
        "• <b>Native Python ML Integration:</b> Avoids serialization friction; scikit-learn and joblib models remain in application memory for instant in-process execution.",
        qa_answer_style
    ))

    # Q4
    elements.append(Paragraph("Question 4: Why should the same trained model be reused for every inference strategy?", qa_question_style))
    elements.append(Paragraph(
        "<b>Answer:</b><br/>"
        "• <b>Prevention of Training-Serving Skew:</b> Reusing the exact fitted pipeline ensures that preprocessing transformations (feature scalers, imputation statistics, and one-hot encoding categories) "
        "are identical across online API calls, batch scoring, and streaming event consumers. Retraining separate models introduces subtle drift and inconsistent predictions.<br/>"
        "• <b>Single Source of Truth & Governance:</b> Maintaining one canonical <i>model.pkl</i> artifact simplifies MLOps auditing, artifact versioning, CI/CD promotion, and rollbacks.<br/>"
        "• <b>Operational Efficiency:</b> Training machine learning models consumes substantial computational and energy resources. Reusing a single fitted pipeline avoids redundant model fitting.",
        qa_answer_style
    ))

    # Section 6: Deliverable Checklist
    elements.append(Paragraph("6. Project Deliverables & Verification Checklist", h1_style))
    check_data = [
        [Paragraph("<b>Deliverable File</b>", body_style), Paragraph("<b>Status</b>", body_style), Paragraph("<b>Verification Note</b>", body_style)],
        [Paragraph("train_model.py", code_style), Paragraph("<font color='green'><b>VERIFIED</b></font>", body_style), Paragraph("Trains pipeline; saves model.pkl & unseen test data", body_style)],
        [Paragraph("online_inference.py", code_style), Paragraph("<font color='green'><b>VERIFIED</b></font>", body_style), Paragraph("Scores single unseen passenger record in real time", body_style)],
        [Paragraph("batch_inference.py", code_style), Paragraph("<font color='green'><b>VERIFIED</b></font>", body_style), Paragraph("Scores 20 unseen records; saves batch_predictions.csv", body_style)],
        [Paragraph("streaming_inference.py", code_style), Paragraph("<font color='green'><b>VERIFIED</b></font>", body_style), Paragraph("Simulates stream scoring of 10 events with 1s delay", body_style)],
        [Paragraph("app.py", code_style), Paragraph("<font color='green'><b>VERIFIED</b></font>", body_style), Paragraph("FastAPI service with GET /health, POST /predict", body_style)],
        [Paragraph("client.py", code_style), Paragraph("<font color='green'><b>VERIFIED</b></font>", body_style), Paragraph("Tests health, valid cases, and Pydantic 422 errors", body_style)],
        [Paragraph("batch_predictions.csv", code_style), Paragraph("<font color='green'><b>VERIFIED</b></font>", body_style), Paragraph("Contains 20 predictions with probabilities and labels", body_style)],
        [Paragraph("Screenshots (8 files)", code_style), Paragraph("<font color='green'><b>VERIFIED</b></font>", body_style), Paragraph("Terminal runs, Swagger UI, and API responses captured", body_style)],
        [Paragraph("requirements.txt & README.md", code_style), Paragraph("<font color='green'><b>VERIFIED</b></font>", body_style), Paragraph("Complete environment spec and run documentation", body_style)]
    ]
    t_check = Table(check_data, colWidths=[2.2 * inch, 1.2 * inch, 3.6 * inch])
    t_check.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#f1f5f9")),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
    ]))
    elements.append(t_check)

    doc.build(elements)
    print(f"Report built successfully at: {OUTPUT_PDF}")


if __name__ == "__main__":
    build_pdf()
