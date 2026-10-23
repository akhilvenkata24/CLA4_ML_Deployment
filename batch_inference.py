"""
CLA 4 - Task 3: Batch Inference
Description:
    Loads the trained model pipeline (model.pkl) and performs batch inference
    on 20 genuine unseen passenger records. Saves all predictions to batch_predictions.csv.
"""

import os
import joblib
import pandas as pd


def main():
    print("=" * 60)
    print("             TASK 3: BATCH INFERENCE (20 RECORDS)")
    print("=" * 60)

    # 1. Path definitions
    model_path = os.path.join(os.path.dirname(__file__), "model.pkl")
    unseen_path = os.path.join(os.path.dirname(__file__), "data", "unseen_test_records.csv")
    output_csv = os.path.join(os.path.dirname(__file__), "batch_predictions.csv")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}. Run train_model.py first.")
    if not os.path.exists(unseen_path):
        raise FileNotFoundError(f"Unseen test data not found: {unseen_path}. Run train_model.py first.")

    # 2. Load trained model pipeline
    print(f"\n[1] Loading saved model from: {model_path}")
    model = joblib.load(model_path)
    print("    Model pipeline loaded successfully.")

    # 3. Load 20 genuine unseen passenger records
    print(f"\n[2] Loading 20 genuine unseen records from: {unseen_path}")
    df_unseen = pd.read_csv(unseen_path)
    batch_records = df_unseen.iloc[0:20].copy()

    feature_cols = ["Pclass", "Age", "SibSp", "Parch", "Fare", "Sex", "Embarked"]
    X_batch = batch_records[feature_cols]

    # 4. Predict all 20 records in a single batch call
    print("\n[3] Executing batch prediction on 20 records simultaneously...")
    predictions = model.predict(X_batch)
    probabilities = model.predict_proba(X_batch)[:, 1]

    # 5. Compile results
    results_df = X_batch.copy()
    results_df["Actual_Survived"] = batch_records["Actual_Survived"].values
    results_df["Predicted_Survived"] = predictions
    results_df["Survival_Probability"] = (probabilities * 100).round(2)
    results_df["Prediction_Label"] = results_df["Predicted_Survived"].map(
        {0: "Did Not Survive", 1: "Survived"}
    )
    results_df["Status"] = (
        results_df["Actual_Survived"] == results_df["Predicted_Survived"]
    ).map({True: "Correct", False: "Mismatch"})

    # 6. Save results to batch_predictions.csv
    results_df.to_csv(output_csv, index=False)
    print(f"\n[4] Saved 20 batch predictions to: {output_csv}")

    # 7. Print summary and table
    print("\n[5] Batch Predictions Preview (first 10 of 20):")
    display_cols = [
        "Pclass", "Sex", "Age", "Fare",
        "Actual_Survived", "Predicted_Survived", "Survival_Probability", "Status"
    ]
    print(results_df[display_cols].head(10).to_string(index=True))

    correct_count = (results_df["Status"] == "Correct").sum()
    print(f"\nBatch Performance Summary:")
    print(f"    - Total Records Processed: {len(results_df)}")
    print(f"    - Correct Predictions    : {correct_count} / {len(results_df)}")
    print(f"    - Batch Accuracy         : {correct_count / len(results_df) * 100:.1f}%")
    print(f"    - Predicted Survived (1) : {(predictions == 1).sum()}")
    print(f"    - Predicted Perished (0) : {(predictions == 0).sum()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
