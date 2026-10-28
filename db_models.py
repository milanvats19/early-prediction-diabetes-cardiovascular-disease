"""
Technology: SQLAlchemy ORM.

Each class below becomes an actual table in PostgreSQL. SQLAlchemy handles
writing the SQL (CREATE TABLE, INSERT, SELECT) for us -- we just describe
the shape of the data as a Python class.

Two separate tables: one per predictor, since they store different inputs.
"""

from sqlalchemy import Column, Integer, Float, String, DateTime, ForeignKey
from sqlalchemy.sql import func
from app.database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class DiabetesRecord(Base):
    __tablename__ = "diabetes_predictions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # inputs
    gender = Column(String)
    age = Column(Float)
    hypertension = Column(Integer)
    heart_disease = Column(Integer)
    smoking_history = Column(String)
    bmi = Column(Float)
    hba1c_level = Column(Float)
    blood_glucose_level = Column(Float)

    # outputs
    prediction = Column(Integer)
    risk_label = Column(String)
    probability = Column(Float)


class HeartRecord(Base):
    __tablename__ = "heart_predictions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # inputs
    age = Column(Integer)
    sex = Column(Integer)
    cp = Column(Integer)
    trestbps = Column(Float)
    chol = Column(Float)
    fbs = Column(Integer)
    restecg = Column(Integer)
    thalach = Column(Float)
    exang = Column(Integer)
    oldpeak = Column(Float)
    slope = Column(Integer)
    ca = Column(Integer)
    thal = Column(Integer)

    # outputs
    prediction = Column(Integer)
    risk_label = Column(String)
    probability = Column(Float)
