# app/callbacks/data_labeling.py
from dash import Input, Output, State, no_update, callback_context
from ..database.session import SessionLocal
from ..database.models import Measurement
import pandas as pd
from dash.exceptions import PreventUpdate

def register_data_labeling_callbacks(app):
    """
    Flow:
      1) 'data-label-dropdown' sets 'save-label-btn' enabled
      2) 'save-label-btn' -> saves label to labels-data, moves to Tab 4
    """

    @app.callback(
        Output("save-label-btn", "disabled"),
        Input("data-label-dropdown", "value"),
        prevent_initial_call=True
    )
    def enable_save_button(label):
        # Enable 'save-label-btn' if a label is selected
        return (label is not None) and (not callback_context.triggered)
    
    @app.callback(
        [
            Output("labels-data", "data"),
            Output("save-alert-message", "is_open"),
            Output("save-alert-message", "children"),
            Output("save-alert-message", "color"),
            Output("tabs", "active_tab", allow_duplicate=True),
            Output("tab-4", "disabled"),
        ],
        [
            Input("save-label-btn", "n_clicks"),
            Input("back-to-visualization-btn", "n_clicks"),
        ],
        [
            State("data-label-dropdown", "value"),
            State("label-notes", "value"),
            State("selected-file-1", "data"),
            State("labels-data", "data"),
            State("tabs", "active_tab"),
            # for review data
            State("review-selected-files", "children"),
            State("review-assigned-labels", "children"),
            State("selected-ball-size", "data"),
            State("operator-id-input", "value"),
        ],
        prevent_initial_call=True
    )
    def handle_label_submit_button_and_back_button(
        save_clicks, 
        back_clicks, 
        label_value, 
        notes, 
        file_path, 
        current_labels, 
        current_tab, 
        files, 
        labels,
        ball_size_id, 
        operator_id
    ):
        ctx = callback_context
        if not ctx.triggered:
            raise PreventUpdate    
        button_id = ctx.triggered[0]["prop_id"].split(".")[0]
        
        labels_data = current_labels.copy() if current_labels else {}
        
        if not file_path:
            raise PreventUpdate
        
        if button_id == "save-label-btn":
            labels_data[file_path] = {
                "file_path": file_path,
                "operator_id": operator_id,
                "ball_size": ball_size_id,
                "label": label_value,
                "notes": notes or "",
                "timestamp": pd.Timestamp.now().isoformat()
            }
            return (
                labels_data,
                True,
                "Label saved successfully",
                "success",
                "tab-4",
                False
            )
            
        if button_id == "back-to-visualization-btn":
            return (
                no_update,
                False,
                "",
                "primary",
                "tab-2",
                no_update
            )
        
        # If none of the above, do nothing
        return no_update, no_update, no_update, no_update, no_update, no_update
