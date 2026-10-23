"""
CLA 4 - Task 2: Online Inference
Description:
    Loads the trained model pipeline (model.pkl) and performs real-time,
    low-latency single-record prediction for a single passenger.
    Uses a genuine unseen record from data/unseen_test_records.csv.
"""

import os
import joblib
import pandas as pd


def main():
    print("=" * 60)
    print("             TASK 2: ONLINE INFERENCE (SINGLE RECORD)")
    print("=" * 60)

    # 1. Path definitions
    model_path = os.path.join(os.path.dirname(__file__), "model.pkl")
    unseen_path = os.path.join(os.path.dirname(__file__), "data", "unseen_test_records.csv")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}. Run train_model.py first.")
    if not os.path.exists(unseen_path):
        raise FileNotFoundError(f"Unseen test data not found: {unseen_path}. Run train_model.py first.")

    # 2. Load trained model pipeline
    print(f"\n[1] Loading saved model from: {model_path}")
    model = joblib.load(model_path)
    print("    Model pipeline loaded successfully.")

    # 3. Load genuine unseen test records
    print(f"\n[2] Loading genuine unseen test data from: {unseen_path}")
    df_unseen = pd.read_csv(unseen_path)

    # Select one genuine unseen passenger record (record #0)
    sample_row = df_unseen.iloc[[0]]
    feature_cols = ["Pclass", "Age", "SibSp", "Parch", "Fare", "Sex", "Embarked"]
    X_single = sample_row[feature_cols]
    actual_label = sample_row["Actual_Survived"].values[0]

    print("\n[3] Incoming Passenger Record (Single Online Request):")
    for col, val in X_single.iloc[0].items():
        print(f"    - {col:<10}: {val}")

    # 4. Perform prediction
    prediction = model.predict(X_single)[0]
    probabilities = model.predict_proba(X_single)[0]
    survival_prob = probabilities[1] * 100

    print("\n[4] Prediction Result:")
    print(f"    - Predicted Class : {prediction} ({'Survived' if prediction == 1 else 'Did Not Survive'})")
    print(f"    - Survival Prob   : {survival_prob:.2f}%")
    print(f"    - Actual Ground Truth: {actual_label} ({'Survived' if actual_label == 1 else 'Did Not Survive'})")
    print(f"    - Match Status    : {'CORRECT' if prediction == actual_label else 'MISMATCH'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
