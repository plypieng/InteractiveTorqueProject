# app/callbacks/data_labeling.py
from dash import Input, Output, State, no_update, callback_context
import dash_bootstrap_components as dbc
from ..database.session import SessionLocal
from ..database.models import Measurement, Feature
import pandas as pd
import logging
import html
from dash.exceptions import PreventUpdate

def register_data_labeling_callbacks(app):
    @app.callback(
        [
            Output("labels-data", "data"),
            Output("save-label-btn", "disabled"),
            Output("save-alert-message", "is_open"),
            Output("save-alert-message", "children"),
            Output("save-alert-message", "color"),
            Output("tab-4", "disabled"),
            Output("tabs", "active_tab", allow_duplicate=True),
            Output("confirmation-modal", "is_open"),
            Output("confirmation-content", "children"),
        ],
        [
            Input("save-label-btn", "n_clicks"),
            Input("back-to-visualization-btn", "n_clicks"),
            Input("confirm-submission", "n_clicks"),
            Input("cancel-submission", "n_clicks"),
        ],
        [
            State("data-label-dropdown", "value"),
            State("label-notes", "value"),
            State("selected-file-1", "data"),
            State("labels-data", "data"),
            State("tabs", "active_tab"),
            State("review-selected-files", "children"),
            State("review-assigned-labels", "children"),
            State("selected-ball-size", "data"),
            State("operator-id-input", "value"),
        ],
        prevent_initial_call=True
    )
    def handle_label_and_submit(
        save_clicks, back_clicks, confirm_clicks, cancel_clicks,
        label_value, notes, file_path, current_labels, current_tab, files, labels,
        ball_size_id, operator_id
    ):
        ctx = callback_context
        if not ctx.triggered:
            raise PreventUpdate
            
        button_id = ctx.triggered[0]["prop_id"].split(".")[0]
        
        # Initialize default return values

        modal_open = False
        modal_content = None
        alert_open = False
        alert_message = ""
        alert_color = "primary"
        new_active_tab = no_update
        tab4_disabled = no_update
        labels_data = current_labels.copy() if current_labels else {}
        
        if not file_path:
            return (
                no_update, True, True, "No file selected", "warning",
                True, no_update, False, None
            )
        
       
        if button_id == "save-label-btn":
            if not label_value:
                return (
                    True, True, "Please select a label", "warning",
                    no_update, no_update, False, None
                )
            # Update labels data
            labels_data[file_path] = {
                "label": label_value,
                "notes": notes or "",
                "timestamp": pd.Timestamp.now().isoformat()
            }
            alert_message = "Label saved successfully"
            alert_color = "success"
            # Enable proceed to Review & Submit
            return (
                labels_data, False, True, alert_message, alert_color,
                False, "tab-4", False, None
            )
        
        if button_id == "back-to-visualization-btn":
            return (
                no_update, False, False, "", "primary",
                no_update, "tab-2", False, None
            )
        
        elif button_id == "confirm-submission":
            try:
                session = SessionLocal()
                
                # Find existing measurement or create new one
                measurement = session.query(Measurement).filter(
                    Measurement.file_path == file_path
                ).first()
                
                if not measurement:
                    measurement = Measurement(
                        file_path=file_path,
                        operator_id=operator_id,
                        ball_size_id=ball_size_id,
                        timestamp=pd.Timestamp.now(),
                        status='Labeled'
                    )
                    session.add(measurement)
                    session.flush()
                
                # Update measurement with label information
                label_info = labels_data.get(file_path, {})
                measurement.label = label_info.get("label")
                measurement.notes = label_info.get("notes", "")
                measurement.status = 'Labeled'
                
                # Add label as a feature for training
                label_feature = Feature(
                    measurement_id=measurement.id,
                    feature_name='Label',
                    feature_value=1.0 if label_info.get("label") == "PASS" else 0.0
                )
                session.add(label_feature)
                
                session.commit()
                alert_message = "Labels and features submitted successfully"
                alert_color = "success"
                new_active_tab = current_tab
                
            except Exception as e:
                logging.error(f"Error submitting labels: {e}")
                alert_message = f"Error submitting labels: {str(e)}"
                alert_color = "danger"
                session.rollback()
            finally:
                session.close()
                
            return (
                labels_data, False, True, alert_message, alert_color,
                False, new_active_tab, False, None
            )
        
        elif button_id == "cancel-submission":
            return (
                labels_data, False, False, "", "primary",
                no_update, no_update, False, None
            )
        
        # Show confirmation modal
        modal_content = html.Div([
            html.H6("Selected Files:", className="mt-3"),
            html.Div(files, className="ms-3"),
            html.H6("Assigned Labels:", className="mt-3"),
            html.Div(labels, className="ms-3"),
        ])
        
        return (
            labels_data, False, False, "", "primary",
            False, no_update, True, modal_content
        )
