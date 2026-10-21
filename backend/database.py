r"""
database.py  --  Milestone M3
Connects FastAPI to PostgreSQL.

Everything in this file is "plumbing":
  * engine     = the connection to the database
  * SessionLocal = a factory that creates one database session per request
  * Base       = the class all our tables inherit from
  * get_db     = a FastAPI "dependency" that opens/closes a session for us

The connection string (DATABASE_URL) is read from the .env file in the
project root, so no passwords are ever written in the code.
"""

import os

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

# Project root = the folder above this file (backend/..)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load .env (DATABASE_URL, SECRET_KEY) from D:\heartRisk\.env
load_dotenv(os.path.join(ROOT, ".env"))

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL is missing - check D:\\heartRisk\\.env")

# pool_pre_ping=True checks the connection before each use, so the server
# survives a database restart without crashing.
engine = create_engine(DATABASE_URL, pool_pre_ping=True)

# Sessions are created fresh for every request and closed afterwards.
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Tables defined in models.py will inherit from this class.
Base = declarative_base()


def get_db():
    """FastAPI dependency: give each request its own database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
