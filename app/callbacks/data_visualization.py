# app/callbacks/data_visualization.py
from dash import Input, Output, State, no_update, dash_table, dcc
from ..processing.data_loader import load_data
from ..processing.filters import high_pass_filter
from ..processing.feature_extraction import extract_features, calculate_rms
from ..processing.measurement_processor import analyze_torque_data

from ..plots.plot_factory import create_normal_plot, create_filtered_plot, create_fft_plot
import plotly.graph_objs as go
import pandas as pd
import logging
import os
import joblib
import shutil

from functools import wraps

from ..config import Config
from ..utils.file_security import is_safe_path
from pdf_generator.pdf_creator import generate_pdf

# Model cache to avoid redundant loading
MODEL_CACHE = {}

def get_cached_model(model_path):
    """Retrieve model from cache or load it if not present."""
    if model_path in MODEL_CACHE:
        return MODEL_CACHE[model_path]
    
    if os.path.exists(model_path):
        try:
            model = joblib.load(model_path)
            MODEL_CACHE[model_path] = model
            logging.info(f"Model loaded and cached: {model_path}")
            return model
        except Exception as e:
            logging.error(f"Error loading model {model_path}: {e}")
            return None
    return None

def register_data_visualization_callbacks(app):
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
            logging.info(f"Back to File Selection button clicked. n_clicks={n_clicks}")
            return "tab-1"
        return no_update

    @app.callback(
        Output("params-offcanvas", "is_open"),
        Input("open-params-canvas", "n_clicks"),
        State("params-offcanvas", "is_open"),
        prevent_initial_call=True
    )
    def toggle_params_offcanvas(n_clicks, is_open):
        if n_clicks:
            return not is_open
        return is_open
    
    @app.callback(
        [
            Output("tabs", "active_tab", allow_duplicate=True),
            Output("tab-3", "disabled"),
            Output("selected-file-name", "children"),
            Output("selected-ball-size-name", "children"),
            Output("selected-operator-name", "children"),
        ],
        Input("proceed-labeling-btn", "n_clicks"),
        [
            State("operator-id-input", "value"),
            State("selected-ball-size", "data"),
            State("selected-file-1", "data"),
        ],
        prevent_initial_call=True
    )
    def switch_to_labeling_tab(n_clicks, operator_id, ball_size_val, file_path_1):
        if n_clicks:
            logging.info(f"Proceed to Labeling button clicked. n_clicks={n_clicks}")
            file_name = os.path.basename(file_path_1)
            return "tab-3", False, file_name, str(ball_size_val), operator_id
        return no_update, no_update, no_update, no_update, no_update
    
    @app.callback(
        [
            Output("normal-graph", "figure"),
            Output("filtered-graph", "figure"),
            Output("fft-graph", "figure"),
            Output("analysis-result", "children"),
            Output("proceed-labeling-btn", "disabled"),
            Output("anomaly-alert", "is_open"),
            Output("anomaly-alert-text", "children"),
            Output("anomaly-alert", "color"),
            Output("model-prediction", "children"),
            Output("analysis-summary", "children"),
            Output("model-prediction-summary", "children"),
            Output("features-data", "data"),
            Output("features-table-2", "children"),
            
            # NEW OUTPUTS for the summary card fields:
            Output("analysis-file-name", "children"),
            Output("analysis-operator-id", "children"),
            Output("analysis-ball-size", "children"),
            Output("analysis-measurement-date", "children"),
            Output("analysis-analysis-date", "children"),
            Output("analysis-size", "children"),
            Output("analysis-number", "children"),
            Output("analysis-rpm", "children"),
            Output("analysis-result-big", "children"),
            Output("analysis-confidence-bar", "value"),
            Output("analysis-confidence-text", "children"),
            Output("analysis-model-info", "data"),
            Output("loading-overlay", "style", allow_duplicate=True),
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
            Input("selected-model", "data"),
        ],
        [   
            State("selected-ball-size", "data"), 
            State("operator-id-input", "value")
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
        selected_model,
        ball_size_id,
        operator_id,
    ):
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
        analysis_summary = ""
        stored_features = {}
        feature_table_2 = no_update
        
        # Validate required inputs
        if not all([file_path_1, cutoff_freq, rms_window_size, hpf_rms_threshold, 
                    spike_threshold, y_axis_range, ball_size_id, operator_id]):
            return (
                normal_fig, 
                filtered_fig, 
                fft_fig,
                "Please fill in all required fields",
                True,
                False,
                "Missing required inputs",
                "warning",
                model_prediction_text,
                analysis_summary,
                model_prediction_text,
                {},
                no_update,
                
                "Unknown",
                "Unknown",
                "Unknown",
                "N/A",
                "N/A",
                "N/A",
                "N/A",
                "N/A",
                "Unknown",
                0,      # Confidence Value
                "N/A",  # Confidence Text
            )

        # Load data
        try:
            data = load_data(file_path_1)
            if data is None or "N[Ncm]" not in data.columns:
                raise ValueError("Invalid data format")
        except Exception as e:
            error_msg = f"Error loading data: {str(e)}"
            logging.error(error_msg)
            return (
                normal_fig, filtered_fig, fft_fig,
                "Error loading data", error_msg,
                True, False, error_msg, "danger",
                model_prediction_text, analysis_summary, model_prediction_text,
                {}, no_update,
                "Unknown", "Unknown", "Unknown", "N/A", "N/A", "N/A", "N/A", "N/A", "Unknown", 0, "N/A"
            )

        # Process file & generate plots
        try:
            figs, extracted_features_dict, result_text = process_file(
                file_path_1,
                cutoff_freq,
                rms_window_size,
                hpf_rms_threshold,
                spike_threshold,
                y_axis_range,
                ball_size_id,
                operator_id,
                file_path_2=file_path_2,
            )
            if not figs or len(figs) != 3:
                raise ValueError("Invalid result from data processing")
            
            normal_fig, filtered_fig, fft_fig = figs
            #features_text = "\n".join([f"{k}: {v}" for k, v in extracted_features_dict.items()])
            analysis_result_text = result_text
            proceed_disabled = "No sudden spike detected in filtered RMS data" not in result_text
            analysis_summary = "PASS" if "PASS" in result_text else "FAIL"

            # store features as global variable dict
            stored_features = extracted_features_dict

            # for basic information
            file_name_display = os.path.basename(file_path_1) if file_path_1 else "Unknown"
            operator_display = operator_id if operator_id else "Unknown"
            ball_size_display = str(ball_size_id) if ball_size_id else "Unknown"
            measurement_date_display = extracted_features_dict.get("Measurement date", "N/A")
            analysis_date_display = extracted_features_dict.get("Analysis Date", "N/A")
            size_display = extracted_features_dict.get("Size", "N/A")
            number_display = extracted_features_dict.get("Number", "N/A")
            rpm_display = extracted_features_dict.get("RPM", "N/A")   
            
            # big pass or fail
            big_result = ""
            confidence_value = 0
            confidence_text = "N/A"
            model_info_text = "Model not loaded"
            
            
            # Build the DataTable for Tab 2
            feature_rows = [{"Feature": k, "Value": v} for k, v in stored_features.items()]
            columns = [{"name": "Feature", "id": "Feature"}, {"name": "Value", "id": "Value"}]
            feature_table_2 = dash_table.DataTable(
                data=feature_rows,
                columns=columns,
                style_as_list_view=True,
                style_table={"overflowX": "auto", "backgroundColor": "#222"},
                style_header={"fontWeight": "bold", 
                              "backgroundColor": "#333", 
                              "color": "white"},
                style_cell={"textAlign": "center",
                            "backgroundColor": "#444",
                            "color": "white",
                            "border": "1px solid #222",
                            "padding": "5px"},
            )
            
            if not selected_model:
                logging.debug("No model selected; skip AI predictions.")
                # We can return early or do partial info
                # For instance:
                return (
                    normal_fig,
                    filtered_fig,
                    fft_fig,
                    "Please select a model in Tab 1",
                    True,    # proceed_disabled
                    False,   # anomaly_is_open
                    "",
                    "success",
                    "No model selected",
                    "",
                    "No model selected",
                    {},
                    no_update,
                    file_name_display,
                    operator_display,
                    ball_size_display,
                    measurement_date_display,
                    analysis_date_display,
                    size_display,
                    number_display,
                    rpm_display,
                    "Unknown",
                    0,
                    "N/A",
                    "N/A",
                    {"display": "none"}
                )

            
            # *** NEW *** Attempt to load your trained pipeline and predict
            try:
                PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
                models_dir = os.path.join(PROJECT_ROOT, "trained_models")
                chosen_model = selected_model  # from the store
                logging.debug(f"DEBUG: selected_model = {chosen_model}")
                model_path = os.path.join(models_dir, str(chosen_model))
                 
                best_model = get_cached_model(model_path)
                if best_model is not None and stored_features:
                    # Convert the feature dict to a DataFrame with same columns as training
                    # Remove non-numeric or metadata columns the model doesn't expect
                    drop_cols = ["Measurement date", "Analysis Date", 
                                 "Ball Size", "Number", "RPM", 
                                 "Size", "Operator"]  # any columns not in training
                    # Safely drop if they exist
                    feature_df = pd.DataFrame([stored_features])
                    feature_df = feature_df.drop(
                        columns=[c for c in drop_cols if c in feature_df.columns],
                        errors="ignore"
                    )
                    
                    # Predict
                    prediction = best_model.predict(feature_df)[0]
                    probability = 0.0
                    if hasattr(best_model, "predict_proba"):
                        probability = best_model.predict_proba(feature_df)[0][1]
                    
                    prediction_label = "◎OK" if prediction == 1 else "✖NG"
                    model_prediction_text = f"Model Prediction: {prediction_label} (Confidence: {probability:.2f})"
                    model_info_text = f"Model version: {chosen_model}"
                    
                    # Override the big_result if you prefer the model's result
                    big_result = f"AI判断結果: {prediction_label}"    
                    confidence_value = probability * 100
                    confidence_text = f"{probability:.2%}"

                    # Save to DB
                    with SessionLocal() as session:
                        # Check if measurement exists
                        existing_measurement = session.query(Measurement).filter_by(
                            file_path=file_path_1,
                            model_version=str(chosen_model)
                        ).first()
                        
                        if existing_measurement:
                            # Update existing measurement
                            existing_measurement.prediction = prediction_label
                            existing_measurement.confidence = probability
                            existing_measurement.prediction_timestamp = datetime.datetime.now()
                            existing_measurement.status = "Predicted"
                        else:
                            # Create new measurement if it doesn't exist
                            new_measurement = Measurement(
                                file_path=file_path_1,
                                operator_id=operator_id,
                                ball_size=float(ball_size_id) if ball_size_id else None,
                                model_version=str(chosen_model),
                                prediction=prediction_label,
                                confidence=probability,
                                prediction_timestamp=datetime.datetime.now(),
                                status="Predicted"
                            )
                            session.add(new_measurement)

                        session.commit()

                else:
                    model_prediction_text = "No valid features for model prediction"
            except Exception as e:
                logging.error(f"Error in model prediction: {e}")
                model_prediction_text = "Error in model prediction or Model not loaded"
                model_info_text = "Model not loaded"

        except Exception as e:
            logging.error(f"Error updating graphs: {e}", exc_info=True)
            return [no_update]*25 + [{"display": "none"}]

        return (
            normal_fig,
            filtered_fig,
            fft_fig,
            analysis_result_text,
            proceed_disabled,
            
            anomaly_is_open,
            anomaly_message,
            anomaly_color,
            model_prediction_text,
            analysis_summary,
            
            model_prediction_text,
            stored_features,
            feature_table_2,
            
            file_name_display,
            operator_display,
            ball_size_display,
            measurement_date_display,
            analysis_date_display,
            size_display,
            number_display,
            rpm_display,
            big_result,
            confidence_value,
            confidence_text,
            model_info_text,
            {"display": "none"} # Hide loading overlay
        )

    # Client-side callback for graph synchronization
    app.clientside_callback(
        """
        function(relayoutData, fig) {
            if (!relayoutData) return window.dash_clientside.no_update;
            if (relayoutData['xaxis.range[0]'] || relayoutData['xaxis.autorange']) {
                let newFig = JSON.parse(JSON.stringify(fig));
                if (!newFig.layout) newFig.layout = {};
                if (!newFig.layout.xaxis) newFig.layout.xaxis = {};
                
                if (relayoutData['xaxis.autorange']) {
                     newFig.layout.xaxis.autorange = true;
                     delete newFig.layout.xaxis.range;
                } else {
                    newFig.layout.xaxis.range = [relayoutData['xaxis.range[0]'], relayoutData['xaxis.range[1]']];
                    newFig.layout.xaxis.autorange = false;
                }
                return newFig;
            }
            return window.dash_clientside.no_update;
        }
        """,
        Output('filtered-graph', 'figure', allow_duplicate=True),
        Input('normal-graph', 'relayoutData'),
        State('filtered-graph', 'figure'),
        prevent_initial_call=True
    )
    
    @app.callback(
        [
            Output("download-pdf", "data"),
            Output("download-alert", "is_open"),
            Output("download-alert", "children"),
            Output("download-alert", "color"),
        ],
        Input("download-pdf-btn", "n_clicks"),
        State("selected-file-1", "data"),
        prevent_initial_call=True
    )
    def download_pdf(n_clicks, file_path):
        if n_clicks:
            logging.info(f"Download PDF button clicked. File path: {file_path}")
            try:
                pdf_path = generate_pdf(file_path)
                logging.info(f"Generated PDF path: {pdf_path}")
                if pdf_path:
                    # Native save to Downloads folder
                    downloads_path = os.path.join(os.path.expanduser("~"), "Downloads")
                    filename = os.path.basename(pdf_path)
                    dest_path = os.path.join(downloads_path, filename)
                    shutil.copy2(pdf_path, dest_path)
                    logging.info(f"File saved to: {dest_path}")
                    return no_update, True, f"PDF saved to Downloads: {filename}", "success"
                else:
                    logging.error("generate_pdf returned None")
                    return no_update, True, "Error generating PDF", "danger"
            except Exception as e:
                logging.error(f"Error in download_pdf: {e}", exc_info=True)
                return no_update, True, f"Error: {str(e)}", "danger"
        return no_update, no_update, no_update, no_update
        
    @app.callback(
        [
            Output("download-csv", "data"),
            Output("download-alert", "is_open", allow_duplicate=True),
            Output("download-alert", "children", allow_duplicate=True),
            Output("download-alert", "color", allow_duplicate=True),
        ],
        Input("download-csv-btn", "n_clicks"),
        State("selected-file-1", "data"),
        prevent_initial_call=True
    )
    def download_csv(n_clicks, file_path):
        if n_clicks and file_path:
            logging.info(f"Download CSV button clicked. File path: {file_path}")
            try:
                # Native save to Downloads folder
                downloads_path = os.path.join(os.path.expanduser("~"), "Downloads")
                filename = os.path.basename(file_path)
                dest_path = os.path.join(downloads_path, filename)
                shutil.copy2(file_path, dest_path)
                logging.info(f"File saved to: {dest_path}")
                return no_update, True, f"CSV saved to Downloads: {filename}", "success"
            except Exception as e:
                logging.error(f"Error in download_csv: {e}", exc_info=True)
                return no_update, True, f"Error: {str(e)}", "danger"
        return no_update, no_update, no_update, no_update


def process_file(
    file_path,
    cutoff_freq,
    rms_window_size,
    hpf_rms_threshold,
    spike_threshold,
    y_axis_range,
    ball_size_id,
    operator_id,
    file_path_2=None,
):
    return analyze_torque_data(
        file_path,
        cutoff_freq,
        rms_window_size,
        hpf_rms_threshold,
        spike_threshold,
        y_axis_range,
        ball_size_id,
        operator_id,
        file_path_2=file_path_2,
    )
