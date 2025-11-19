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
from .database.session import engine

def apply_extra_migrations():
    # Ensure order_id column exists in measurements
    insp = inspect(engine)
    if insp.has_table('measurements'):
        cols = [col['name'] for col in insp.get_columns('measurements')]
        if 'order_id' not in cols:
            with engine.connect() as conn:
                conn.execute(text('ALTER TABLE measurements ADD COLUMN order_id TEXT'))

def create_app():
    # Apply database migrations at startup
    run_migrations.run_migrations()
    # Apply ad-hoc schema change for order_id
    apply_extra_migrations()
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
    
    # Initialize Dash app
    app = Dash(
        __name__,
        external_stylesheets=[
            dbc.themes.CYBORG,
            'https://use.fontawesome.com/releases/v5.15.4/css/all.css'
        ],
        suppress_callback_exceptions=True,  # Allow callbacks for dynamically loaded components (e.g. Tabs)
        meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
    )
    
    app.layout = create_layout()
    
    register_callbacks(app)
    config_callbacks.register_config_callbacks(app)
    
    return app
