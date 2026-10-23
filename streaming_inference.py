"""
CLA 4 - Task 4: Streaming Inference
Description:
    Simulates a streaming data pipeline (such as Kafka or EventBridge).
    Processes 10 genuine unseen passenger records sequentially, simulating
    event arrival with a 1-second delay between messages.
"""

import os
import time
import joblib
import pandas as pd


def main():
    print("=" * 65)
    print("        TASK 4: STREAMING INFERENCE (10 SEQUENTIAL EVENTS)")
    print("=" * 65)

    # 1. Path definitions
    model_path = os.path.join(os.path.dirname(__file__), "model.pkl")
    unseen_path = os.path.join(os.path.dirname(__file__), "data", "unseen_test_records.csv")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}. Run train_model.py first.")
    if not os.path.exists(unseen_path):
        raise FileNotFoundError(f"Unseen test data not found: {unseen_path}. Run train_model.py first.")

    # 2. Load trained model pipeline once
    print(f"\n[1] Loading saved model from: {model_path}")
    model = joblib.load(model_path)
    print("    Model pipeline loaded successfully.")

    # 3. Load genuine unseen test records
    print(f"\n[2] Loading stream source from: {unseen_path}")
    df_unseen = pd.read_csv(unseen_path)
    stream_records = df_unseen.iloc[20:30].copy()  # Distinct records 20-29

    feature_cols = ["Pclass", "Age", "SibSp", "Parch", "Fare", "Sex", "Embarked"]

    print(f"\n[3] Starting real-time stream consumer...")
    print(f"    Processing 10 passenger events with 1.0 second delay between records...\n")
    print(f"{'Event #':<8} | {'Passenger Features':<35} | {'Prediction':<17} | {'Survival Prob':<14} | {'Status'}")
    print("-" * 88)

    correct_count = 0
    total_events = len(stream_records)

    for i, (_, row) in enumerate(stream_records.iterrows(), start=1):
        # Simulate incoming stream latency
        time.sleep(1.0)

        # Single record slice
        X_event = pd.DataFrame([row[feature_cols]])
        actual = int(row["Actual_Survived"])

        # Predict event
        pred = int(model.predict(X_event)[0])
        prob = model.predict_proba(X_event)[0][1] * 100

        is_correct = (pred == actual)
        if is_correct:
            correct_count += 1

        feature_summary = f"P{int(row['Pclass'])}, {row['Sex']}, Age:{row['Age'] if pd.notnull(row['Age']) else 'NaN'}, Fare:{row['Fare']:.1f}"
        pred_label = "Survived (1)" if pred == 1 else "Not Survived (0)"
        status_label = "CORRECT" if is_correct else "MISMATCH"

        print(f"Event {i:<3} | {feature_summary:<35} | {pred_label:<17} | {prob:>6.2f}%       | {status_label}")

    print("-" * 88)
    print(f"\nStream Processing Complete:")
    print(f"    - Total Events Streamed: {total_events}")
    print(f"    - Correct Predictions  : {correct_count} / {total_events}")
    print(f"    - Stream Accuracy      : {correct_count / total_events * 100:.1f}%")
    print(f"    - Total Stream Elapsed : ~{total_events} seconds")
    print("=" * 65)


if __name__ == "__main__":
    main()
