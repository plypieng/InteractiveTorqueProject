# app/database/session.py
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from ..config import Config
import os

Base = declarative_base()

print("DEBUG: final DB path = ", os.path.abspath("torque_data.db"))
engine = create_engine(Config.DATABASE_URL, echo=False)
SessionLocal = sessionmaker(bind=engine)

def init_db():
    """
    Call this function to explicitly create all tables.
    Example usage: in 'initialize_db.py'
    """
    Base.metadata.create_all(engine)
