# celery_worker.py
from celery import Celery
import os

# Initialize Celery
celery_app = Celery(
    'tasks',
    broker='redis://localhost:6379/0',  # Ensure Redis is running
    backend='redis://localhost:6379/0'
)

# Optional: Configure Celery settings as needed
celery_app.conf.update(
    result_expires=3600,
)

# Import tasks after Celery app initialization to avoid circular imports
from app.tasks import measurement_tasks
