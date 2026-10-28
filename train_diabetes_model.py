"""
Trains a diabetes risk prediction model on a GENERAL POPULATION dataset
(~100,000 patients, both genders) rather than the old Pima Indians dataset,
which only contained women.

Columns: gender, age, hypertension, heart_disease, smoking_history, bmi,
         HbA1c_level, blood_glucose_level, diabetes (target)

Technology used here:
- pandas -> load & clean the CSV
- pd.get_dummies -> one-hot encodes the two text columns (gender,
  smoking_history) into numeric 0/1 columns the model can use
- scikit-learn -> train/test split, RandomForestClassifier, evaluation
- joblib -> save the trained model + scaler + final feature column list
  (the column list matters a lot here because one-hot encoding changes
  the number/order of columns, and the API must reproduce it exactly)
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # backend/
DATA_PATH = os.path.join(BASE_DIR, "data", "diabetes.csv")
MODEL_DIR = os.path.join(BASE_DIR, "app", "models")

CATEGORICAL_COLS = ["gender", "smoking_history"]


def load_and_encode_data():
    df = pd.read_csv(DATA_PATH)
    df = pd.get_dummies(df, columns=CATEGORICAL_COLS)
    return df


def train():
    df = load_and_encode_data()

    X = df.drop("diabetes", axis=1)
    y = df["diabetes"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=3,
        class_weight="balanced",  # dataset is ~91.5% non-diabetic / 8.5% diabetic
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    print("=== Diabetes Model Evaluation (general population dataset) ===")
    print(f"Accuracy : {accuracy_score(y_test, y_pred):.3f}")
    print(f"Precision: {precision_score(y_test, y_pred):.3f}")
    print(f"Recall   : {recall_score(y_test, y_pred):.3f}  <-- most important: catching true positives")
    print(f"F1 Score : {f1_score(y_test, y_pred):.3f}")
    print(f"ROC-AUC  : {roc_auc_score(y_test, y_proba):.3f}")

    importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    print("\nFeature importance:")
    print(importances)

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, os.path.join(MODEL_DIR, "diabetes_model.pkl"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "diabetes_scaler.pkl"))
    joblib.dump(list(X.columns), os.path.join(MODEL_DIR, "diabetes_features.pkl"))
    print(f"\nSaved model to {MODEL_DIR}")


if __name__ == "__main__":
    train()
