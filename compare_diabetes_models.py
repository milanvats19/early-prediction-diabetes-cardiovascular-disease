"""
Trains and compares Logistic Regression, Random Forest, and XGBoost on the
general-population diabetes dataset, and saves whichever performs best.

Note: this dataset has 100,000 rows, so GridSearchCV grids here are kept
smaller than you might use on a tiny dataset -- otherwise this script would
take a very long time to run. Feel free to widen the grids if you have time
and want to squeeze out more performance.

Run it with:  python3 -m app.compare_diabetes_models
"""

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from xgboost import XGBClassifier
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


def evaluate(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return {
        "model": name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }


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

    results = []
    fitted_models = {}

    # ---------- 1. Logistic Regression ----------
    lr_grid = GridSearchCV(
        LogisticRegression(max_iter=2000, class_weight="balanced"),
        param_grid={"C": [0.1, 1, 10]},
        scoring="roc_auc",
        cv=3,
    )
    lr_grid.fit(X_train_scaled, y_train)
    fitted_models["LogisticRegression"] = lr_grid.best_estimator_
    results.append(evaluate("LogisticRegression", lr_grid.best_estimator_, X_test_scaled, y_test))
    print(f"LogisticRegression best params: {lr_grid.best_params_}")

    # ---------- 2. Random Forest ----------
    rf_grid = GridSearchCV(
        RandomForestClassifier(class_weight="balanced", random_state=42, n_jobs=-1),
        param_grid={
            "n_estimators": [200, 300],
            "max_depth": [8, 12],
        },
        scoring="roc_auc",
        cv=3,
    )
    rf_grid.fit(X_train_scaled, y_train)
    fitted_models["RandomForest"] = rf_grid.best_estimator_
    results.append(evaluate("RandomForest", rf_grid.best_estimator_, X_test_scaled, y_test))
    print(f"RandomForest best params: {rf_grid.best_params_}")

    # ---------- 3. XGBoost ----------
    pos = (y_train == 1).sum()
    neg = (y_train == 0).sum()
    xgb_grid = GridSearchCV(
        XGBClassifier(
            eval_metric="logloss",
            scale_pos_weight=neg / pos,
            random_state=42,
            n_jobs=-1,
        ),
        param_grid={
            "n_estimators": [200, 300],
            "max_depth": [4, 6],
        },
        scoring="roc_auc",
        cv=3,
    )
    xgb_grid.fit(X_train_scaled, y_train)
    fitted_models["XGBoost"] = xgb_grid.best_estimator_
    results.append(evaluate("XGBoost", xgb_grid.best_estimator_, X_test_scaled, y_test))
    print(f"XGBoost best params: {xgb_grid.best_params_}")

    # ---------- Compare and pick a winner ----------
    results_df = pd.DataFrame(results).set_index("model")
    print("\n=== Diabetes model comparison (general population dataset) ===")
    print(results_df.round(3))

    best_name = results_df.sort_values(["roc_auc", "recall"], ascending=False).index[0]
    best_model = fitted_models[best_name]
    print(f"\nBest model: {best_name}")

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(best_model, os.path.join(MODEL_DIR, "diabetes_model.pkl"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "diabetes_scaler.pkl"))
    joblib.dump(list(X.columns), os.path.join(MODEL_DIR, "diabetes_features.pkl"))
    results_df.round(3).to_csv(os.path.join(MODEL_DIR, "diabetes_model_comparison.csv"))
    print(f"\nSaved best model ({best_name}) to {MODEL_DIR}")


if __name__ == "__main__":
    train()
