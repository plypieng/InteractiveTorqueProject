# app/callbacks/data_visualization.py
from dash import Input, Output, State, no_update, callback_context, html, dcc
import dash_bootstrap_components as dbc
from ..processing.data_loader import load_data
from ..processing.filters import high_pass_filter
from ..processing.feature_extraction import extract_all_features, calculate_fft, calculate_rms
from ..processing.anomaly_detection import detected_sudden_spike, analyse_hpf_rms, detect_anomalous_measurement
from ..plots.plot_factory import create_normal_plot, create_filtered_plot, create_fft_plot
from ..database.session import SessionLocal
from ..database.models import Measurement, Feature, BallSize
import plotly.graph_objs as go
import pandas as pd
import logging
import os
import joblib
import numpy as np
from functools import wraps
from dash.exceptions import PreventUpdate
from ..config import Config
from ..utils.file_security import is_safe_path
from pdf_generator.pdf_creator import generate_pdf

def register_data_visualization_callbacks(app):
    # Define a decorator for logging callback errors including duplicates
    def log_callback_errors(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if "Duplicate callback outputs" in str(e):
                    logging.warning(f"Duplicate callback detected in {func.__name__}: {e}")
                else:
                    logging.error(f"Error in callback '{func.__name__}': {e}", exc_info=True)
                return no_update
        return wrapper
    
    @app.callback(
            Output("tabs", "active_tab", allow_duplicate=True),
            Input("back-to-file-selection-btn", "n_clicks"),
            prevent_initial_call=True
    )
    def switch_to_file_selection_tab(n_clicks):
        if n_clicks:
            logging.info("Switching to file selection tab") 
            return "tab-1"
        return no_update
    
    @app.callback(
            Output("tabs", "active_tab", allow_duplicate=True),
            Output("tab-3", "disabled"),
            Output("selected-file-name", "children"),
            Output("selected-ball-size-name", "children"),
            Output("selected-operator-name", "children"),
            Input("proceed-labeling-btn", "n_clicks"),
            Input("operator-id-input", "value"),
            Input("ball-size-dropdown", "value"),
            Input("selected-file-1", "data"),
            prevent_initial_call=True,
    )
    def switch_to_labeling_tab(n_clicks, operator_id, ball_size_id, file_path_1):
        if n_clicks:
            logging.info("Switching to labeling tab")
            file_name = os.path.basename(file_path_1)

            return "tab-3", False, file_name, ball_size_id , operator_id
        return no_update
    
    @app.callback(
        [
            Output("normal-graph", "figure"),
            Output("filtered-graph", "figure"),
            Output("fft-graph", "figure"),
            Output("features", "children"),
            Output("analysis-result", "children"),
            Output("proceed-labeling-btn", "disabled"),
            Output("anomaly-alert", "is_open"),
            Output("anomaly-alert-text", "children"),
            Output("anomaly-alert", "color"),
            Output("model-prediction", "children"),
            Output("analysis-summary", "children"),
            Output("model-prediction-summary", "children"),
        ],
        [
            Input("selected-file-1", "data"),
            Input("selected-file-2", "data"),
            Input("cutoff-input", "value"),
            Input("rms-window-size", "value"),
            Input("hpf-rms-threshold", "value"),
            Input("spike-threshold", "value"),
            Input("y-axis-slider", "value"),
            Input("initial-torque-input", "value"),
        ],
        [
            State("selected-ball-size", "data"),
            State("operator-id-input", "value"),
        ],
        prevent_initial_call=True
    )
    @log_callback_errors
    def update_graphs(
        file_path_1,
        file_path_2,
        cutoff_freq,
        rms_window_size,
        hpf_rms_threshold,
        spike_threshold,
        y_axis_range,
        initial_torque,
        ball_size_id,
        operator_id,
    ):
        # Initialize default values
        normal_fig = go.Figure()
        filtered_fig = go.Figure()
        fft_fig = go.Figure()
        features_text = ""
        analysis_result_text = ""
        proceed_disabled = True  
        anomaly_is_open = False
        anomaly_message = ""
        anomaly_color = "success"
        model_prediction_text = "Model not loaded"
        session = None
        analysis_summary = ""
        

        # Validate required inputs
        if not all([file_path_1, cutoff_freq, rms_window_size, hpf_rms_threshold, 
                    spike_threshold, y_axis_range, ball_size_id, operator_id]):
            return (
                normal_fig, filtered_fig, fft_fig,
                "Missing required inputs", "Please fill in all required fields",
                True, False, "Missing required inputs", "warning", model_prediction_text, analysis_summary, model_prediction_text,
            )

        try:
            # Create database session
            session = SessionLocal()

            # Load and validate data
            try:
                data = load_data(file_path_1)
                if data is None or "N[Ncm]" not in data.columns:
                    raise ValueError("Invalid data format")
                torque_values = data["N[Ncm]"].to_numpy()
            except Exception as e:
                error_msg = f"Error loading data: {str(e)}"
                logging.error(error_msg)
                return (
                    normal_fig, filtered_fig, fft_fig,
                    "Error loading data", error_msg,
                    True, False, error_msg, "danger", model_prediction_text, analysis_summary, model_prediction_text
                )

            # Perform anomaly detection
            #try:
            #    anomaly, message = detect_anomalous_measurement(torque_values, ball_size_id, session)
            #    if anomaly:
            #        return (
            #            normal_fig, filtered_fig, fft_fig,
            #            "", f"ANOMALY DETECTED: {message}",
            #            True, True, message, "danger", model_prediction_text
            #        )
            #except Exception as e:
            #    error_msg = f"Error in anomaly detection: {str(e)}"
            #    logging.error(error_msg)
            #    return (
            #        normal_fig, filtered_fig, fft_fig,
            #        "", error_msg,
            #        True, False, error_msg, "danger", model_prediction_text
            #    )

            # Process file and generate plots
            try:
                figs, features_text, analysis_result_text = process_file(
                    file_path_1, cutoff_freq, rms_window_size,
                    hpf_rms_threshold, spike_threshold, y_axis_range,
                    "1", ball_size_id, operator_id
                )
                
                if not figs or len(figs) != 3:
                    raise ValueError("Invalid result from data processing")
                
                normal_fig, filtered_fig, fft_fig = figs
                proceed_disabled = "No sudden spike detected in filtered RMS data" not in analysis_result_text

                analysis_summary = "PASS" if "PASS" in analysis_result_text else "FAIL"
            
        

                # Perform model inference
                try:
                    best_model = joblib.load("best_model.pkl")  # Ensure the model is loaded correctly
                    if best_model is not None:
                        measurement = session.query(Measurement).filter(
                            Measurement.file_path == file_path_1
                        ).first()
                        
                        if measurement and measurement.features:
                            feature_dict = {
                                feature.feature_name: feature.feature_value 
                                for feature in measurement.features 
                                if feature.feature_name != 'Label'
                            }
                            feature_df = pd.DataFrame([feature_dict])
                            
                            prediction = best_model.predict(feature_df)[0]
                            probability = (
                                best_model.predict_proba(feature_df)[0][1] 
                                if hasattr(best_model, 'predict_proba') else 0.0
                            )
                            
                            prediction_label = "PASS" if prediction == 1 else "FAIL"
                            model_prediction_text = (
                                f"Model Prediction: {prediction_label} "
                                f"(Confidence: {probability:.2f})"
                            )
                        else:
                            model_prediction_text = "No features found for prediction"
                except Exception as e:
                    logging.error(f"Error in model prediction: {e}")
                    model_prediction_text = "Error in model prediction or Model is not loaded"

            except Exception as e:
                error_msg = f"Error processing file: {str(e)}"
                logging.error(error_msg)
                return (
                    normal_fig, filtered_fig, fft_fig,
                    "", error_msg,
                    True, False, error_msg, "danger", model_prediction_text, analysis_summary, model_prediction_text
                )

        except Exception as e:
            error_msg = f"Error updating graphs: {str(e)}"
            logging.error(error_msg)
            return (
                normal_fig, filtered_fig, fft_fig,
                "", error_msg,
                True, False, error_msg, "danger", model_prediction_text, analysis_summary, model_prediction_text
            )
        finally:
            if session is not None:
                session.close()

        # Ensure the return is outside the try-except-finally blocks
        return (
            normal_fig,
            filtered_fig,
            fft_fig,
            features_text,
            analysis_result_text,
            proceed_disabled,
            anomaly_is_open,
            anomaly_message,
            anomaly_color,
            model_prediction_text,
            analysis_summary,
            model_prediction_text
        )
    
    #handle pdf download button call back
    @app.callback(
        Output("download-pdf", "data"),
        Input("download-pdf-btn", "n_clicks"),
        State("selected-file-1", "data"),
        prevent_initial_call=True
    )
    def download_pdf(n_clicks, file_path):
        #if clicked generate pdf
        if n_clicks:
            pdf_path = generate_pdf(file_path)
            return dcc.send_file(pdf_path)
        
    #handle csv download button call back
    @app.callback(
        Output("download-csv", "data"),
        Input("download-csv-btn", "n_clicks"),
        State("selected-file-1", "data"),
        prevent_initial_call=True
    )
    def download_csv(n_clicks, file_path):
        if n_clicks:
            logging.info(f"Download CSV button clicked with filepath='{file_path}'")
            csv_path = file_path
            return dcc.send_file(csv_path)

def process_file(
    file_path,
    cutoff_freq,
    rms_window_size,
    hpf_rms_threshold,
    spike_threshold,
    y_axis_range,
    file_label,
    ball_size_id,
    operator_id,
):
    from dash import no_update
    from dash.exceptions import PreventUpdate
    from ..processing.data_loader import load_data
    from ..processing.filters import high_pass_filter
    from ..processing.feature_extraction import extract_all_features, calculate_fft, calculate_rms
    from ..processing.anomaly_detection import detected_sudden_spike, analyse_hpf_rms
    from ..plots.plot_factory import create_normal_plot, create_filtered_plot, create_fft_plot
    from ..database.session import SessionLocal
    from ..database.models import Measurement, Feature
    import pandas as pd
    import numpy as np
    import logging
    import plotly.graph_objs as go

    # Initialize variables
    normal_fig = go.Figure()
    filtered_fig = go.Figure()
    fft_fig = go.Figure()
    features = {}
    features_text = ""
    analysis_result_text = ""
    db_session = None

    # Security check
    if not is_safe_path(Config.ALLOWED_DIRECTORY, file_path):
        error_msg = "Invalid file path selected."
        logging.warning(f"Attempt to access invalid file path: {file_path}")
        error_fig = go.Figure()
        error_fig.add_annotation(
            text=error_msg,
            xref="paper",
            yref="paper",
            showarrow=False,
            font=dict(color="red", size=16),
        )
        return [error_fig, go.Figure(), go.Figure()], "", error_msg

    try:
        # Read data and validate required columns
        data = load_data(file_path)
        if data is None:
            raise ValueError("Failed to load data from file")

        required_columns = ["X[mm]", "N[Ncm]"]
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")

        # Extract data columns
        x = data["X[mm]"].copy()
        y = data["N[Ncm]"].copy()
        
        if len(x) == 0 or len(y) == 0:
            raise ValueError("Data is empty")
            
        if y.isnull().any() or x.isnull().any():
            raise ValueError("Data contains invalid values (NaN)")
        
        # Extract features
        features = extract_all_features(y, cutoff=cutoff_freq)
        if not features:
            raise ValueError("Failed to extract features")

        # High-Pass Filtering
        y_filtered = high_pass_filter(y.to_numpy(), cutoff=cutoff_freq)
        if y_filtered is None:
            raise ValueError("Failed to apply high-pass filter")

        filtered_series = pd.Series(y_filtered)
        filtered_rms = filtered_series.rolling(window=rms_window_size).apply(calculate_rms, raw=True)
        if filtered_rms.isnull().all():
            raise ValueError("Failed to calculate RMS")
        
        # Compute moving max and min of the filtered data
        moving_max = filtered_series.rolling(window=100, min_periods=1).max()
        moving_min = filtered_series.rolling(window=100, min_periods=1).min()
        
        # Compute moving average of the moving max and min
        moving_max_avg = moving_max.rolling(window=int(rms_window_size), min_periods=1).mean()
        moving_min_avg = moving_min.rolling(window=int(rms_window_size), min_periods=1).mean()

        # FFT
        xf, amplitudes = calculate_fft(y.to_numpy())
        if xf is None or amplitudes is None:
            raise ValueError("Failed to calculate FFT")

        # Create plots
        normal_fig = create_normal_plot(x, y, y_axis_range)
        filtered_fig = create_filtered_plot(
            x, y_filtered, filtered_rms, moving_max_avg, moving_min_avg, cutoff_freq, y_axis_range
        )
        fft_fig = create_fft_plot(xf, amplitudes)

        # Generate analysis result
        analysis_result = []
        spike_detected = detected_sudden_spike(filtered_rms, spike_threshold)
        if spike_detected:
            analysis_result.append(
                "Sudden spike detected in filtered RMS data. Re-measurement recommended."
            )
        else:
            analysis_result.append("No sudden spike detected in filtered RMS data.")
        
        hpf_rms_result = analyse_hpf_rms(filtered_rms, hpf_rms_threshold)
        analysis_result.append(hpf_rms_result)
        
        if spike_detected or "over the threshold" in hpf_rms_result.lower():
            overall_result = "FAILED"
        else:
            overall_result = "PASSED"
        
        analysis_result_text = "\n".join(analysis_result)
        analysis_result_text = f"Overall analysis result: {overall_result}\n{analysis_result_text}"

        # Format output text
        operator_id = operator_id if operator_id else "Unknown Operator"
        timestamp = pd.Timestamp.now().isoformat()
        features_text = "\n".join([f"{key}: {value}" for key, value in features.items()])
        features_text += f"\nOperator ID: {operator_id}\nTimestamp: {timestamp}"

        # Store measurement and features in the database
        try:
            db_session = SessionLocal()
            measurement = Measurement(
                file_path=file_path,
                operator_id=operator_id,
                ball_size_id=ball_size_id,
                timestamp=pd.Timestamp.now(),
                status='Processed'
            )
            db_session.add(measurement)
            db_session.flush()
            
            feature_objects = [
                Feature(
                    measurement_id=measurement.id,
                    feature_name=name,
                    feature_value=value
                )
                for name, value in features.items()
            ]
            db_session.bulk_save_objects(feature_objects)
            db_session.commit()
        except Exception as e:
            if db_session is not None:
                db_session.rollback()
            logging.error(f"Error storing measurement and features: {e}")
            # Continue even if database storage fails
        finally:
            if db_session is not None:
                db_session.close()
        
    except Exception as e:
        logging.error(f"Error processing file: {e}")
        return [go.Figure(), go.Figure(), go.Figure()], "", ""
    
    # Ensure the return is outside the try-except-finally blocks
    return (
        [normal_fig,
        filtered_fig,
        fft_fig],
        features_text,
        analysis_result_text
    )


