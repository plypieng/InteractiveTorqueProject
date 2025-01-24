# app/__init__.py
from dash import Dash
import dash_bootstrap_components as dbc
from .ui.layout import create_layout
from .callbacks import register_callbacks
from .config import Config
import logging
import os

def create_app():
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
        suppress_callback_exceptions=False,  # Catch duplicate callbacks
        meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
    )
    
    app.layout = create_layout()
    
    register_callbacks(app)
    
    return app
