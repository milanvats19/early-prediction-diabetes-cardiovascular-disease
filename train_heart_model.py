"""
Trains a cardiovascular disease (heart disease) risk prediction model.

Dataset: UCI Heart Disease (Cleveland), commonly used in ML tutorials.
Columns: age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang,
         oldpeak, slope, ca, thal, target(0/1)

Technology used here (same pattern as train_diabetes_model.py so the team
only has to learn one workflow):
- pandas -> load data
- scikit-learn -> train/test split, RandomForestClassifier, evaluation
- joblib -> persist trained model + scaler + feature list
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # backend/
DATA_PATH = os.path.join(BASE_DIR, "data", "heart.csv")
MODEL_DIR = os.path.join(BASE_DIR, "app", "models")


def load_data():
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")  # handles BOM in the file
    df.columns = [c.strip() for c in df.columns]
    return df


def train():
    df = load_data()

    X = df.drop("target", axis=1)
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=6,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=42,
    )
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    print("=== Heart Disease (CVD) Model Evaluation ===")
    print(f"Accuracy : {accuracy_score(y_test, y_pred):.3f}")
    print(f"Precision: {precision_score(y_test, y_pred):.3f}")
    print(f"Recall   : {recall_score(y_test, y_pred):.3f}  <-- most important: catching true positives")
    print(f"F1 Score : {f1_score(y_test, y_pred):.3f}")
    print(f"ROC-AUC  : {roc_auc_score(y_test, y_proba):.3f}")

    importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    print("\nFeature importance:")
    print(importances)

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, os.path.join(MODEL_DIR, "heart_model.pkl"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "heart_scaler.pkl"))
    joblib.dump(list(X.columns), os.path.join(MODEL_DIR, "heart_features.pkl"))
    print(f"\nSaved model to {MODEL_DIR}")


if __name__ == "__main__":
    train()
