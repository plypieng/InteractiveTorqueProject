# app/callbacks/data_labeling.py
from dash import Input, Output, State, no_update, callback_context
from ..database.session import SessionLocal
from ..database.models import Measurement, AuditLog
import pandas as pd
from dash.exceptions import PreventUpdate
import datetime
import logging

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
            State("ball-size-input", "value"), # Changed from selected-ball-size to ball-size-input
            State("operator-id-input", "value"),
            State("selected-model", "data"), # Changed from model-name-store
        ],
        prevent_initial_call=True
    )
    def handle_label_submit_button_and_back_button(
        save_n_clicks, 
        back_n_clicks, 
        label_value, 
        notes_value, 
        file_path, 
        labels_data,           # State 6
        active_tab,            # State 7
        review_files,          # State 8
        review_labels,         # State 9
        selected_ball_size,    # State 10
        ball_size_val,         # State 11 (ball-size-input)
        operator_id,           # State 12
        model_name             # State 13
    ):
        ctx = callback_context # Changed from dash.callback_context
        if not ctx.triggered:
            return no_update, no_update, no_update, no_update, no_update, no_update

        button_id = ctx.triggered[0]["prop_id"].split(".")[0]

        if button_id == "back-to-visualization-btn":
            if label_value and not labels_data: # Label selected but not saved
                return no_update, True, "Warning: Unsaved label. Are you sure you want to go back?", "warning", no_update, no_update
            return no_update, no_update, no_update, no_update, "tab-2", no_update

        if button_id == "save-label-btn":
            if not file_path or not label_value:
                return no_update, no_update, no_update, no_update, no_update, no_update

            try:
                # Save to DB
                with SessionLocal() as session:
                    # Check if measurement exists
                    existing_measurement = session.query(Measurement).filter_by(
                        file_path=file_path,
                        model_version=str(model_name)
                    ).first()
                    
                    if existing_measurement:
                        # Capture previous state
                        prev_label = existing_measurement.label
                        prev_status = existing_measurement.status
                        
                        existing_measurement.label = label_value
                        existing_measurement.notes = notes_value
                        existing_measurement.operator_id = operator_id
                        existing_measurement.ball_size = float(ball_size_val) if ball_size_val else None
                        existing_measurement.submitted_timestamp = datetime.datetime.now()
                        new_status = "Labeled"
                        existing_measurement.status = new_status
                        
                        # Create Audit Log if changed
                        if prev_label != label_value or prev_status != new_status:
                            audit = AuditLog(
                                measurement_id=existing_measurement.id,
                                changed_by=operator_id,
                                previous_status=prev_status,
                                new_status=new_status,
                                previous_label=prev_label,
                                new_label=label_value,
                                change_reason="Manual Labeling Update"
                            )
                            session.add(audit)
                    else:
                        # Create new record if not exists
                        start_time = datetime.datetime.now() # Fallback
                        new_meas = Measurement(
                            file_path=file_path,
                            label=label_value,
                            notes=notes_value,
                            operator_id=operator_id,
                            ball_size=float(ball_size_val) if ball_size_val else None,
                            model_version=str(model_name),
                            submitted_timestamp=start_time,
                            measurement_time=start_time, # Defaulting to now if we don't have it
                            analysis_time=start_time,
                            status="Labeled"
                        )
                        session.add(new_meas)
                        session.flush() # Get ID
                        
                        # Initial Audit Log
                        audit = AuditLog(
                            measurement_id=new_meas.id,
                            changed_by=operator_id,
                            previous_status=None,
                            new_status="Labeled",
                            previous_label=None,
                            new_label=label_value,
                            change_reason="Initial Labeling"
                        )
                        session.add(audit)

                    session.commit()
                
                # Update local store
                new_data = {
                    "file": file_path, 
                    "label": label_value, 
                    "notes": notes_value,
                    "ball_size": ball_size_val,
                    "operator": operator_id
                }
                
                return new_data, True, "Label saved successfully", "success", "tab-4", False
            
            except Exception as e:
                logging.error(f"Error saving label: {e}", exc_info=True)
                return no_update, True, f"Error saving label: {e}", "danger", no_update, no_update
        
        # If none of the above, do nothing
        return no_update, no_update, no_update, no_update, no_update, no_update

    @app.callback(
        [
            Output("data-label-dropdown", "value"),
            Output("save-label-btn", "n_clicks"),
        ],
        Input("keyboard-event", "data"),
        [
            State("save-label-btn", "n_clicks"),
            State("save-label-btn", "disabled"),
            State("tabs", "active_tab"),
        ],
        prevent_initial_call=True
    )
    def handle_keyboard_shortcuts(event_data, current_n_clicks, is_disabled, active_tab):
        if not event_data or active_tab != "tab-3":
            return no_update, no_update
        
        key = event_data.get("key")
        
        if key == "p":
            return "PASS", no_update
        elif key == "f":
            return "FAIL", no_update
        elif key == "enter":
            if not is_disabled:
                return no_update, (current_n_clicks or 0) + 1
            
        return no_update, no_update
        return no_update, no_update
