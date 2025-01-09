# app/callbacks/review_submission.py
from dash import Input, Output, State, no_update
import dash_bootstrap_components as dbc
from ..database.session import SessionLocal
from ..database.models import Measurement, Feature
import pandas as pd
import logging
from dash.exceptions import PreventUpdate

def register_review_submission_callbacks(app):
    @app.callback(
        [
            Output("review-selected-files", "children"),
            Output("review-assigned-labels", "children"),
            Output("submit-labels-btn", "disabled"),
        ],
        [
            Input("labels-data", "data"),
            Input("tabs", "active_tab"),
        ],
        [
            State("selected-file-1", "data"),
        ],
        prevent_initial_call=True
    )
    def update_review_display_and_submit(labels_data, active_tab, file_path):
        if not file_path or not labels_data or file_path not in labels_data:
            return dbc.Alert("No files selected"), dbc.Alert("No labels assigned"), True
        
        label_info = labels_data[file_path]
        file_name = os.path.basename(file_path)
        
        files_list = html.Ul([
            html.Li([
                html.I(className="fas fa-file me-2"),
                file_name
            ], className="mb-2")
        ], className="list-unstyled")
        
        labels_list = html.Ul([
            html.Li([
                html.Strong("Label: "),
                html.Span(label_info["label"], className="ms-2")
            ], className="mb-2"),
            html.Li([
                html.Strong("Notes: "),
                html.Span(label_info["notes"] or "No notes provided", className="ms-2")
            ], className="mb-2"),
            html.Li([
                html.Strong("Timestamp: "),
                html.Span(label_info["timestamp"], className="ms-2")
            ])
        ], className="list-unstyled")
        
        # Enable submit button only on review tab with valid labels
        submit_disabled = not (active_tab == "tab-4" and labels_data and len(labels_data) > 0)
        
        return files_list, labels_list, submit_disabled
