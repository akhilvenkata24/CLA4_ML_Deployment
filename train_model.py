"""
CLA 4 - Phase 1: Model Training Pipeline
Dataset: Titanic Kaggle Competition (data/train.csv)
Description:
    Trains a classification pipeline to predict passenger survival.
    Handles numerical and categorical preprocessing with imputation and encoding.
    Saves the entire fitted pipeline as 'model.pkl' and extracts genuine unseen
    records for downstream inference tasks.
"""

import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


def main():
    print("=" * 60)
    print("        TITANIC SURVIVAL MODEL TRAINING PIPELINE")
    print("=" * 60)

    # 1. Dataset Loading & Inspection
    data_path = os.path.join(os.path.dirname(__file__), "data", "train.csv")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Training data not found at: {data_path}")

    print(f"\n[1] Loading dataset from: {data_path}")
    df = pd.read_csv(data_path)
    print(f"    Raw records loaded: {df.shape[0]} rows, {df.shape[1]} columns")

    # Target and Feature Definitions
    target_col = "Survived"
    numerical_features = ["Pclass", "Age", "SibSp", "Parch", "Fare"]
    categorical_features = ["Sex", "Embarked"]
    feature_cols = numerical_features + categorical_features

    print(f"    Target column: '{target_col}'")
    print(f"    Numerical features:   {numerical_features}")
    print(f"    Categorical features: {categorical_features}")
    print(f"    Excluded columns:     ['PassengerId', 'Name', 'Ticket', 'Cabin']")

    X = df[feature_cols]
    y = df[target_col]

    # 2. Train / Test Split (Stratified 80/20)
    print("\n[2] Splitting dataset into training and testing portions...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    print(f"    Training records: {len(X_train)} ({len(X_train)/len(df)*100:.1f}%)")
    print(f"    Testing records:  {len(X_test)} ({len(X_test)/len(df)*100:.1f}%)")

    # Save genuine unseen test records for inference strategies
    unseen_df = X_test.copy()
    unseen_df["Actual_Survived"] = y_test.values
    unseen_path = os.path.join(os.path.dirname(__file__), "data", "unseen_test_records.csv")
    unseen_df.to_csv(unseen_path, index=False)
    print(f"    Saved {len(unseen_df)} genuine unseen test records to: {unseen_path}")

    # 3. Preprocessing Pipelines
    print("\n[3] Building preprocessing and classification pipeline...")
    numerical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numerical_transformer, numerical_features),
        ("cat", categorical_transformer, categorical_features)
    ])

    # 4. Model Definition & Pipeline Assembly
    classifier = RandomForestClassifier(
        n_estimators=100,
        max_depth=6,
        random_state=42
    )

    model_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", classifier)
    ])

    # 5. Model Training (Single fitting)
    print("\n[4] Fitting complete pipeline on training data...")
    model_pipeline.fit(X_train, y_train)
    print("    Training completed successfully.")

    # 6. Model Evaluation on Unseen Test Set
    print("\n[5] Evaluating model on testing set...")
    y_pred = model_pipeline.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    print(f"    Actual Test Accuracy: {test_accuracy * 100:.2f}%")
    print("\n--- Detailed Classification Report ---")
    print(classification_report(y_test, y_pred, target_names=["Not Survived (0)", "Survived (1)"]))

    # 7. Save Pipeline to model.pkl
    model_path = os.path.join(os.path.dirname(__file__), "model.pkl")
    joblib.dump(model_pipeline, model_path)
    print(f"\n[6] Saved complete fitted pipeline to: {model_path}")
    print(f"    Model file size: {os.path.getsize(model_path) / 1024:.2f} KB")

    # 8. Verification: Reload model.pkl and predict a test record
    print("\n[7] Verifying saved model by reloading and predicting a test record...")
    loaded_model = joblib.load(model_path)
    sample_record = X_test.iloc[[0]]
    actual_label = y_test.iloc[0]
    predicted_label = loaded_model.predict(sample_record)[0]
    predicted_prob = loaded_model.predict_proba(sample_record)[0]

    print(f"    Sample Record:\n{sample_record.to_dict(orient='records')[0]}")
    print(f"    Actual Label:    {actual_label} ({'Survived' if actual_label == 1 else 'Not Survived'})")
    print(f"    Predicted Label: {predicted_label} ({'Survived' if predicted_label == 1 else 'Not Survived'})")
    print(f"    Survival Probability: {predicted_prob[1]*100:.2f}%")
    print(f"    Model reload and prediction verification: PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
