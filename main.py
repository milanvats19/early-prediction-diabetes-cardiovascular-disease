"""
Technology: FastAPI (Python web framework) + Uvicorn (the server that runs it).

FastAPI is used here because:
- It automatically validates incoming JSON against our Pydantic schemas
- It generates free interactive API docs at /docs
- It's async-ready and fast, which matters if this ever needs to scale

This file is what your Backend Developer teammate owns and extends
(e.g. adding a database, user accounts, saving prediction history).
"""

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from sqlalchemy import desc
import joblib
import os
import numpy as np
import pandas as pd

from app.schemas import DiabetesInput, HeartInput, PredictionOutput, UserSignup, UserLogin, Token, UserOut
from app.database import Base, engine, get_db
from app.db_models import DiabetesRecord, HeartRecord, User
from app.auth import hash_password, verify_password, create_access_token, get_current_user

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

app = FastAPI(
    title="Diabetes & Cardiovascular Disease Risk Prediction API",
    description="Takes health metrics and returns an early risk prediction. "
                 "This is a decision-support tool, NOT a medical diagnosis.",
    version="1.0.0",
)

# CORS: allows the frontend (running on a different port/domain) to call this API.
# For development we allow all origins; for production, restrict this to your
# actual frontend domain.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create the database tables (if they don't already exist) as soon as the
# API starts up. Safe to run every time -- it does nothing if tables exist.
Base.metadata.create_all(bind=engine)

# ---- Load trained models once at startup (fast predictions, no retraining per request) ----
diabetes_model = joblib.load(os.path.join(MODEL_DIR, "diabetes_model.pkl"))
diabetes_scaler = joblib.load(os.path.join(MODEL_DIR, "diabetes_scaler.pkl"))
diabetes_features = joblib.load(os.path.join(MODEL_DIR, "diabetes_features.pkl"))

heart_model = joblib.load(os.path.join(MODEL_DIR, "heart_model.pkl"))
heart_scaler = joblib.load(os.path.join(MODEL_DIR, "heart_scaler.pkl"))
heart_features = joblib.load(os.path.join(MODEL_DIR, "heart_features.pkl"))


def risk_label_from_probability(prob: float) -> str:
    if prob < 0.33:
        return "Low Risk"
    elif prob < 0.66:
        return "Moderate Risk"
    else:
        return "High Risk"


def top_features(model, feature_names, n=3):
    importances = model.feature_importances_
    idx = np.argsort(importances)[::-1][:n]
    return [feature_names[i] for i in idx]


def encode_diabetes_input(data: DiabetesInput) -> pd.DataFrame:
    """
    The diabetes model was trained on one-hot encoded columns (e.g.
    'gender_Female', 'smoking_history_never'). This rebuilds that same
    encoding for a single incoming request so it exactly matches what
    the model expects, in the same column order it was trained on.
    """
    raw = {
        "age": data.age,
        "hypertension": data.hypertension,
        "heart_disease": data.heart_disease,
        "bmi": data.bmi,
        "HbA1c_level": data.HbA1c_level,
        "blood_glucose_level": data.blood_glucose_level,
        "gender": data.gender.value,
        "smoking_history": data.smoking_history.value,
    }
    df = pd.DataFrame([raw])
    df = pd.get_dummies(df, columns=["gender", "smoking_history"])
    df = df.reindex(columns=diabetes_features, fill_value=0)
    return df


@app.get("/")
def root():
    return {
        "message": "Diabetes & CVD Risk Prediction API is running.",
        "docs": "/docs",
        "endpoints": ["/auth/signup", "/auth/login", "/predict/diabetes", "/predict/heart"],
    }


# ---------------------------------------------------------------
# Auth endpoints
# ---------------------------------------------------------------
@app.post("/auth/signup", response_model=Token)
def signup(data: UserSignup, db: Session = Depends(get_db)):
    existing = db.query(User).filter(User.email == data.email).first()
    if existing:
        raise HTTPException(status_code=400, detail="An account with this email already exists.")

    user = User(email=data.email, hashed_password=hash_password(data.password))
    db.add(user)
    db.commit()
    db.refresh(user)

    token = create_access_token(user.id)
    return Token(access_token=token, user=UserOut.model_validate(user))


@app.post("/auth/login", response_model=Token)
def login(data: UserLogin, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == data.email).first()
    if not user or not verify_password(data.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Incorrect email or password.")

    token = create_access_token(user.id)
    return Token(access_token=token, user=UserOut.model_validate(user))


@app.get("/auth/me", response_model=UserOut)
def read_me(current_user: User = Depends(get_current_user)):
    return current_user


@app.post("/predict/diabetes", response_model=PredictionOutput)
def predict_diabetes(
    data: DiabetesInput,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    try:
        row = encode_diabetes_input(data)
        scaled = diabetes_scaler.transform(row)
        prediction = int(diabetes_model.predict(scaled)[0])
        probability = float(diabetes_model.predict_proba(scaled)[0][1])
        risk_label = risk_label_from_probability(probability)

        # Save this prediction to the database, tied to the logged-in user
        record = DiabetesRecord(
            user_id=current_user.id,
            gender=data.gender.value,
            age=data.age,
            hypertension=data.hypertension,
            heart_disease=data.heart_disease,
            smoking_history=data.smoking_history.value,
            bmi=data.bmi,
            hba1c_level=data.HbA1c_level,
            blood_glucose_level=data.blood_glucose_level,
            prediction=prediction,
            risk_label=risk_label,
            probability=round(probability, 3),
        )
        db.add(record)
        db.commit()

        return PredictionOutput(
            prediction=prediction,
            risk_label=risk_label,
            probability=round(probability, 3),
            top_factors=top_features(diabetes_model, diabetes_features),
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/history/diabetes")
def diabetes_history(
    limit: int = 20,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    records = (
        db.query(DiabetesRecord)
        .filter(DiabetesRecord.user_id == current_user.id)
        .order_by(desc(DiabetesRecord.created_at))
        .limit(limit)
        .all()
    )
    return records


@app.post("/predict/heart", response_model=PredictionOutput)
def predict_heart(
    data: HeartInput,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    try:
        row = [[getattr(data, f) for f in heart_features]]
        scaled = heart_scaler.transform(row)
        prediction = int(heart_model.predict(scaled)[0])
        probability = float(heart_model.predict_proba(scaled)[0][1])
        risk_label = risk_label_from_probability(probability)

        record = HeartRecord(
            user_id=current_user.id,
            **data.model_dump(),
            prediction=prediction,
            risk_label=risk_label,
            probability=round(probability, 3),
        )
        db.add(record)
        db.commit()

        return PredictionOutput(
            prediction=prediction,
            risk_label=risk_label,
            probability=round(probability, 3),
            top_factors=top_features(heart_model, heart_features),
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/history/heart")
def heart_history(
    limit: int = 20,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    records = (
        db.query(HeartRecord)
        .filter(HeartRecord.user_id == current_user.id)
        .order_by(desc(HeartRecord.created_at))
        .limit(limit)
        .all()
    )
    return records
