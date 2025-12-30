# app/__init__.py
from dash import Dash
import dash_bootstrap_components as dbc
from .ui.layout import create_layout
from .callbacks import register_callbacks
from .callbacks import config_callbacks
from .config import Config
import logging
import os
import run_migrations
from sqlalchemy import inspect, text
from .database.session import engine, init_db
from .database import models # Register models
from .database import models # Register models
from .services.file_watcher import FileWatcherService
from dash import DiskcacheManager
import diskcache

def apply_extra_migrations():
    # Ensure order_id column exists in measurements
    insp = inspect(engine)
    if insp.has_table('measurements'):
        cols = [col['name'] for col in insp.get_columns('measurements')]
        if 'order_id' not in cols:
            with engine.connect() as conn:
                conn.execute(text('ALTER TABLE measurements ADD COLUMN order_id TEXT'))

def create_app():
    # Apply database migrations at startup (non-blocking - allow app to start if migrations fail)
    try:
        success = run_migrations.run_migrations()
        if not success:
            logging.warning("Database migrations may have failed or are already current. Continuing anyway.")
    except Exception as e:
        logging.error(f"Error running migrations: {e}. Continuing anyway.",exc_info=True)
    
    # Apply ad-hoc schema change for order_id
    try:
        apply_extra_migrations()
    except Exception as e:
        logging.error(f"Error applying extra migrations: {e}", exc_info=True)
    
    # Ensure all tables exist (including new AuditLog)
    init_db()
    
    # Configure logging
    log_directory = os.path.dirname(os.path.abspath(__file__))
    log_file = os.path.join(log_directory, "app.log")
    
    logging.basicConfig(
        level=logging.DEBUG,  # Capture DEBUG and above
        format="%(asctime)s %(levelname)s:%(message)s",
        handlers=[
            logging.FileHandler(log_file, mode='a', encoding='utf-8'),
            logging.StreamHandler()  # Log to console as well
        ]
    )
    
    # Initialize DiskcacheManager for background callbacks
    cache = diskcache.Cache("./cache")
    background_callback_manager = DiskcacheManager(cache)

    # Initialize Dash app
    app = Dash(
        __name__,
        external_stylesheets=[
            dbc.themes.CYBORG,
            'https://use.fontawesome.com/releases/v5.15.4/css/all.css'
        ],
        suppress_callback_exceptions=True,  # Allow callbacks for dynamically loaded components (e.g. Tabs)
        meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
        background_callback_manager=background_callback_manager
    )
    
    # Start File Watcher
    watcher = FileWatcherService()
    if Config.ALLOWED_DIRECTORY and os.path.exists(Config.ALLOWED_DIRECTORY):
        watcher.start_watching(Config.ALLOWED_DIRECTORY, extension=".csv", cache_key="csv_data")
    
    app.layout = create_layout()
    
    register_callbacks(app)
    config_callbacks.register_config_callbacks(app)
    
    return app
