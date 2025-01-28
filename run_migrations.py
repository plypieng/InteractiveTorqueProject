# run_migrations.py
import os
import sys
from alembic.config import Config
from alembic import command
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_migrations():
    try:
        # Create an Alembic configuration and run the migration
        alembic_cfg = Config("migrations/alembic.ini")
        
        # Stamp the database with the current head revision
        command.stamp(alembic_cfg, "head")
        
        # Run the migration
        command.upgrade(alembic_cfg, "head")
        
        logger.info("Database migration completed successfully!")
    except Exception as e:
        logger.error(f"Error during migration: {e}")
        sys.exit(1)

if __name__ == "__main__":
    run_migrations()