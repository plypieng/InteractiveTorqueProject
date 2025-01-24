# initialize_db.py
import sys
import os
import logging

# Ensure Python sees 'app' as a package
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from app.database.session import init_db
from app.database.models import BallSize, Measurement, Feature

logging.basicConfig(level=logging.INFO)

def main():
    logging.info("Initializing the database...")
    init_db()
    logging.info("Database tables created successfully.")

if __name__ == "__main__":
    main()
