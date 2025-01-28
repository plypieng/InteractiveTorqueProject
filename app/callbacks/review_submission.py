# app/callbacks/review_submission.py
from dash import Input, Output, State, no_update, callback_context, html, dash_table
import dash_bootstrap_components as dbc
from ..database.session import SessionLocal
from ..database.models import Measurement
import pandas as pd
import logging
import os
import re
from dash.exceptions import PreventUpdate
import datetime
def parse_measurement_time(time_str):
        """
        Parse measurement time from format "YYYYMMDDHHMMSS" to datetime object
        Example: "20250110145764" -> datetime(2025, 01, 10, 14, 57, 64)
        """
        try:
            if not time_str or not isinstance(time_str, str):
                return None
            
            # Extract components
            year = int(time_str[0:4])
            month = int(time_str[4:6])
            day = int(time_str[6:8])
            hour = int(time_str[8:10])
            minute = int(time_str[10:12])
            second = int(time_str[12:14])
            
            return datetime.datetime(year, month, day, hour, minute, second)
        except (ValueError, TypeError, IndexError) as e:
            logging.error(f"Error parsing measurement time {time_str}: {e}")
            return datetime.datetime.now()  # Fallback to current time
        
        
def register_review_submission_callbacks(app):

    @app.callback(
        Output("features-table-4", "children"),
        Input("features-data", "data"),
        State("tabs", "active_tab"),
        prevent_initial_call=True
    )
    def update_features_table_4(features_dict, active_tab):
        """
        Show features in tab-4 as a table, for final review.
        """
        if active_tab != "tab-4" or not features_dict:
            return "No features available for review."
        
        feature_rows = [{"Feature": k, "Value": v} for k, v in features_dict.items()]
        columns = [{"name": "Feature", "id": "Feature"}, {"name": "Value", "id": "Value"}]
        table_4 = dash_table.DataTable(
            data=feature_rows,
            columns=columns,
            style_table={"overflowX": "auto"},
            style_cell={"textAlign": "left"},
            style_header={"fontWeight": "bold"},
        )
        return table_4    

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
            State("operator-id-input", "value"),
            State("selected-ball-size", "data"),
            State("features-data", "data")
        ],
        prevent_initial_call=True
    )
    def update_review_display_and_submit(
        labels_data, 
        active_tab, 
        file_path,
        operator_id,
        ball_size,
        features_dict
    ):
        """
        Show file + label summary in Tab 4.
        If no file or no label, disable 'submit-labels-btn'.
        """
        if not file_path or not labels_data or file_path not in labels_data:
            return dbc.Alert("No files selected"), dbc.Alert("No labels assigned"), True
        
        label_info = labels_data[file_path]
        file_name = os.path.basename(file_path)
        
        files_list = html.Ul([
            html.Li([html.I(className="fas fa-file me-2"), file_name], className="mb-2"),
            html.Li([html.Strong("Operator ID: "), html.Span(label_info["operator_id"], className="ms-2")]),
            html.Li([html.Strong("Ball Size: "), html.Span(label_info["ball_size"], className="ms-2")])
        ], className="list-unstyled")
        
        labels_list = html.Ul([
            html.Li([html.Strong("Label: "), html.Span(label_info["label"], className="ms-2")], className="mb-2"),
            html.Li([html.Strong("Notes: "), html.Span(label_info["notes"] or "No notes provided", className="ms-2")], className="mb-2"),
            html.Li([html.Strong("Timestamp: "), html.Span(label_info["timestamp"], className="ms-2")])
        ], className="list-unstyled")

        submit_disabled = not (active_tab == "tab-4" and labels_data and len(labels_data) > 0)
        
        return files_list, labels_list, submit_disabled
    
    @app.callback(
        [
            Output("confirmation-modal", "is_open", allow_duplicate=True),
            Output("confirmation-content", "children")
        ],
        [Input("submit-labels-btn", "n_clicks")],
        [
            State("review-selected-files", "children"),
            State("review-assigned-labels", "children"),
            State("confirmation-modal", "is_open"),
        ],
        prevent_initial_call=True
    )
    def open_confirmation_modal(n_clicks, selected_files, assigned_labels, is_open):
        if not n_clicks:
            raise PreventUpdate

        # Show a simple modal asking for confirmation
        modal_content = html.Div([
            html.P("Are you sure you want to submit these labels?"),
        ])
        return True, modal_content

    
    @app.callback(
        Output("confirmation-modal", "is_open"),
        [
            Input("confirm-submission-btn", "n_clicks"),
            Input("cancel-submission-btn", "n_clicks"),
        ],
        [
            State("confirmation-modal", "is_open"),
            State("selected-file-1", "data"),
            State("labels-data", "data"),
            State("features-data", "data"),
            State("model-prediction", "children"),  # or "model_prediction_text"
            State("selected-model", "data"),
          
        ],
        prevent_initial_call=True
    )
    def finalize_submission(confirm_click, 
                            cancel_click, 
                            is_open, 
                            file_path, 
                            labels_data, 
                            features_data, 
                            model_prediction_text,
                            selected_model
    ):
        """
        Finalize submission: Save everything to DB in a wide-table format, close modal.
        """
        ctx = callback_context
        if not ctx.triggered:
            raise PreventUpdate

        button_id = ctx.triggered[0]["prop_id"].split(".")[0]
        if button_id == "confirm-submission-btn":
            try:
                label_info = labels_data.get(file_path, {})
                with SessionLocal() as session:
                    # find or create measurement
                    measurement = session.query(Measurement).filter_by(
                            file_path=file_path,
                        model_version=selected_model
                    ).first()
                    if not measurement:
                         # Convert string dates to datetime objects
                        measurement_time = parse_measurement_time(features_data.get("Measurement date"))
                        analysis_time = pd.to_datetime(features_data.get("Analysis Date"))
                        submitted_time = pd.Timestamp.now()
                        measurement = Measurement(
                            file_path=file_path,
                            operator_id=label_info.get("operator_id"),
                            ball_size_id=label_info.get("ball_size"),
                            submitted_timestamp=submitted_time.to_pydatetime(),
                            measurement_time=measurement_time,
                            analysis_time=analysis_time.to_pydatetime(),
                            status="submitted",
                            model_version=selected_model,
                        )
                        session.add(measurement)
                        session.flush()

                    # Fill label & notes
                    measurement.label = label_info.get("label")
                    measurement.notes = label_info.get("notes")

                    # Store features in wide columns
                    if features_data:
                        measurement.mean = features_data.get("Mean")
                        measurement.median = features_data.get("Median")
                        measurement.mad = features_data.get("MAD")
                        measurement.standard_deviation = features_data.get("Standard Deviation")
                        measurement.rms = features_data.get("RMS")
                        measurement.shape_factor = features_data.get("Shape Factor")
                        measurement.crest_factor = features_data.get("Crest Factor")
                        measurement.entropy = features_data.get("Entropy")
                        measurement.skewness = features_data.get("Skewness")
                        measurement.kurtosis = features_data.get("Kurtosis")
                        measurement.gradient_mean = features_data.get("Gradient Mean")
                        measurement.gradient_std_dev = features_data.get("Gradient Std Dev")
                        measurement.rolling_median_mean = features_data.get("Rolling Median Mean")
                        measurement.rolling_mad_mean = features_data.get("Rolling MAD Mean")
                        measurement.spectral_centroid = features_data.get("Spectral Centroid")
                        measurement.spectral_entropy = features_data.get("Spectral Entropy")
                        measurement.peak_frequency = features_data.get("Peak Frequency")
                        measurement.spectral_flatness = features_data.get("Spectral Flatness")
                        measurement.spectral_spread = features_data.get("Spectral Spread")
                        measurement.spectral_roll_off = features_data.get("Spectral Roll-off")
                        measurement.low_band_energy = features_data.get("Low Band Energy")
                        measurement.mid_band_energy = features_data.get("Mid Band Energy")
                        measurement.high_band_energy = features_data.get("High Band Energy")
                        measurement.spectral_crest = features_data.get("Spectral Crest")
                        measurement.spectral_flux = features_data.get("Spectral Flux")
                        measurement.spectral_kurtosis = features_data.get("Spectral Kurtosis")
                        measurement.spectral_skewness = features_data.get("Spectral Skewness")
                        measurement.spectral_slope = features_data.get("Spectral Slope")
                        if model_prediction_text and "Confidence:" in model_prediction_text:
                            # naive parse
                            
                            match = re.search(r"(◎OK|✖NG).*\(Confidence:\s*([\d.]+)\)", model_prediction_text)
                            if match:
                                predicted_label = match.group(1)  # e.g. ◎OK or ✖NG
                                confidence_str = match.group(2)   # e.g. 0.95
                                measurement.predicted_label = predicted_label
                                measurement.prediction_confidence = float(confidence_str)
                        
                    # Store model prediction

                    session.commit()
                    
            except Exception as e:
                logging.error(f"Error finalizing submission: {e}", exc_info=True)
                return no_update
            else:
                # After success, close the modal and alert the user
                return not is_open 
            finally:
                # Also close the modal
                return not is_open

        elif button_id == "cancel-submission-btn":
            return not is_open
        else:
            return is_open
    
    
    #callback for back to labeling button    
    @app.callback(
        Output("tabs", "active_tab"),
        Input("back-to-labeling-btn", "n_clicks"),
        prevent_initial_call=True
    )
    def back_to_labeling(n_clicks):
        return "tab-3"
