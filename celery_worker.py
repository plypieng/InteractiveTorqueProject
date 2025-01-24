# celery_worker.py
from celery import Celery

celery_app = Celery(
    'tasks',
    broker='redis://localhost:6379/0',  # Make sure Redis is running on this host/port
    backend='redis://localhost:6379/0'
)

celery_app.conf.update(result_expires=3600)

# Import tasks after Celery app init to avoid circular imports
from app.tasks import measurement_tasks
