"""
Technology: SQLAlchemy (Python's most widely used database toolkit/ORM)
+ psycopg2 (the actual driver that talks to PostgreSQL).

This file sets up:
- the database connection (engine)
- a session factory (SessionLocal) -- each API request gets its own
  short-lived session to talk to the database
- Base -- the class that all our table models (in db_models.py) inherit from

The connection string is read from an environment variable so you never
hardcode passwords into the code. If it's not set, it falls back to a
sensible local default matching the setup instructions in the README.
"""

import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql+psycopg2://postgres:postgres@localhost:5432/diabetes_cvd_db",
)

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    """FastAPI dependency: gives each request its own DB session and
    always closes it afterward, even if an error happens."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
