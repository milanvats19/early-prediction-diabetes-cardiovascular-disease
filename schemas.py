"""
Technology: Pydantic (comes bundled with FastAPI).
Pydantic defines the "shape" of data going in/out of the API and
automatically validates it (e.g. rejects a request if 'age' is missing
or is a string instead of a number) and generates interactive docs.
"""

from pydantic import BaseModel, Field, EmailStr
from enum import Enum


class UserSignup(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=6, description="At least 6 characters")


class UserLogin(BaseModel):
    email: EmailStr
    password: str


class UserOut(BaseModel):
    id: int
    email: str

    class Config:
        from_attributes = True


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserOut


class Gender(str, Enum):
    female = "Female"
    male = "Male"
    other = "Other"


class SmokingHistory(str, Enum):
    never = "never"
    no_info = "No Info"
    current = "current"
    former = "former"
    not_current = "not current"
    ever = "ever"


class DiabetesInput(BaseModel):
    gender: Gender = Field(..., example="Female")
    age: float = Field(..., ge=1, le=120, example=45)
    hypertension: int = Field(..., ge=0, le=1, description="1 = has hypertension", example=0)
    heart_disease: int = Field(..., ge=0, le=1, description="1 = has heart disease", example=0)
    smoking_history: SmokingHistory = Field(..., example="never")
    bmi: float = Field(..., ge=10, le=80, example=27.5)
    HbA1c_level: float = Field(..., ge=3, le=15, description="average blood sugar over 2-3 months", example=5.7)
    blood_glucose_level: float = Field(..., ge=50, le=400, example=120)


class HeartInput(BaseModel):
    age: int = Field(..., ge=1, le=120, example=54)
    sex: int = Field(..., ge=0, le=1, description="1 = male, 0 = female", example=1)
    cp: int = Field(..., ge=0, le=3, description="chest pain type (0-3)", example=0)
    trestbps: float = Field(..., ge=0, le=250, description="resting blood pressure", example=130)
    chol: float = Field(..., ge=0, le=700, description="serum cholesterol mg/dl", example=246)
    fbs: int = Field(..., ge=0, le=1, description="fasting blood sugar > 120 mg/dl (1=true)", example=0)
    restecg: int = Field(..., ge=0, le=2, example=1)
    thalach: float = Field(..., ge=0, le=250, description="max heart rate achieved", example=150)
    exang: int = Field(..., ge=0, le=1, description="exercise induced angina", example=0)
    oldpeak: float = Field(..., ge=0, le=10, example=1.0)
    slope: int = Field(..., ge=0, le=2, example=1)
    ca: int = Field(..., ge=0, le=4, description="number of major vessels colored", example=0)
    thal: int = Field(..., ge=0, le=3, example=2)


class PredictionOutput(BaseModel):
    prediction: int          # 0 = low risk, 1 = high risk
    risk_label: str          # human-readable label
    probability: float       # model confidence (0-1)
    top_factors: list[str]   # which inputs most influenced this type of prediction
