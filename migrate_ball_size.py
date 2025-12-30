import logging
from sqlalchemy import create_engine, text, inspect
from app.database.session import engine

def migrate():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting migration: Adding ball_size column...")
    
    insp = inspect(engine)
    columns = [col['name'] for col in insp.get_columns('measurements')]
    
    if 'ball_size' not in columns:
        try:
            with engine.connect() as conn:
                conn.execute(text("ALTER TABLE measurements ADD COLUMN ball_size FLOAT"))
                logger.info("Added 'ball_size' column successfully.")
        except Exception as e:
            logger.error(f"Error adding column: {e}")
    else:
        logger.info("'ball_size' column already exists.")

    # We are NOT dropping ball_size_id or ball_sizes table to avoid data loss risks with SQLite limitations
    # and to keep it simple. The code will just ignore them.
    
    logger.info("Migration completed.")

if __name__ == "__main__":
    migrate()
