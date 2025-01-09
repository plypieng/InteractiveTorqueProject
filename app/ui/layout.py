# app/ui/layout.py
import dash_bootstrap_components as dbc
from dash import html, dcc
from .components import (
    get_loading_overlay,
    get_confirmation_modal,
    get_progress_indicator,
    get_help_modal,
    get_tab1_content,
    get_tab2_content,
    get_tab3_content,
    get_tab4_content
)

def create_layout():
    return dbc.Container([
        # Overlays and Modals
        get_loading_overlay(),
        get_confirmation_modal(),
        get_help_modal(),
        
        # Header with Title and Help Button
        dbc.Row([
            dbc.Col([
                html.H1("トルクデーター解析 V2", className="text-center text-primary mb-2"),
                html.P("Torque Data Analysis System", className="text-center text-muted mb-4"),
            ], width=12),
            dbc.Col([
                dbc.Button([
                    html.I(className="fas fa-question-circle me-2"),
                    "Help Guide"
                ], id="open-modal", color="info", size="sm", className="mb-3"),
            ], width=12, className="text-center"),
        ]),
        
        # Progress Indicator
        get_progress_indicator(),
        
        # Interval Component for Periodic Updates
        dcc.Interval(
            id="interval-component",
            interval=10*1000,  # 10 seconds
            n_intervals=0,
        ),
        
        # Hidden Stores for State Management
        dcc.Store(id="selected-file-1"),
        dcc.Store(id="selected-file-2"),
        dcc.Store(id="selected-ball-size"),
        dcc.Store(id="labels-data", data={}),
        dcc.Store(id="prev-file-list", data=[]),
        
        # Main Tabs
        dbc.Tabs([
            dbc.Tab(label="File Selection", tab_id="tab-1", id="tab-1", children=[
                get_tab1_content()
            ]),
            dbc.Tab(label="Data Visualization", tab_id="tab-2", id="tab-2", children=[
                get_tab2_content()
            ], disabled=True),
            dbc.Tab(label="Data Labeling", tab_id="tab-3", id="tab-3", children=[
                get_tab3_content()
            ], disabled=True),
            dbc.Tab(label="Review & Submit", tab_id="tab-4", id="tab-4", children=[
                get_tab4_content()
            ], disabled=True),
        ], id="tabs", active_tab="tab-1"),
    ], fluid=True)
