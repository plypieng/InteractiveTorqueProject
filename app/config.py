# app/config.py
import os
from dotenv import load_dotenv

# Load environment variables from .env file, if any
load_dotenv()

class Config:
    LOG_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
    LOG_FILE = os.path.join(LOG_DIRECTORY, "app.log")
    ALLOWED_DIRECTORY = os.getenv('ALLOWED_DIRECTORY', 'W:\\')  # Example default if not set
    DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///torque_data.db')
    CELERY_BROKER_URL = os.getenv('CELERY_BROKER_URL', 'redis://localhost:6379/0')
    CELERY_BACKEND_URL = os.getenv('CELERY_BACKEND_URL', 'redis://localhost:6379/0')
    # Add more config parameters as needed
