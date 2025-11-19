# app/callbacks/data_visualization.py
from dash import Input, Output, State, no_update, dash_table
from ..processing.data_loader import load_data
from ..processing.filters import high_pass_filter
from ..processing.feature_extraction import extract_features, calculate_rms

from ..plots.plot_factory import create_normal_plot, create_filtered_plot, create_fft_plot
import plotly.graph_objs as go
import pandas as pd
import logging
import os
import joblib

from functools import wraps

from ..config import Config
from ..utils.file_security import is_safe_path
from pdf_generator.pdf_creator import generate_pdf



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
            logging.info("Switching to file selection tab") 
            return "tab-1"
        return no_update
    
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
            State("ball-size-dropdown", "value"),
            State("selected-file-1", "data"),
        ],
        prevent_initial_call=True
    )
    def switch_to_labeling_tab(n_clicks, operator_id, ball_size_id, file_path_1):
        if n_clicks:
            logging.info("Switching to labeling tab")
            file_name = os.path.basename(file_path_1)
            return "tab-3", False, file_name, ball_size_id, operator_id
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
            Output("analysis-confidence", "children"),
            Output("analysis-model-info", "data"),
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
                "N/A",
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
                {}, no_update
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
            # will be parse from model_prediction_text later when the model is implemented
            confidence_display = "(Confidence: )"
            
            
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
                    "Unknown",
                    "Unknown",
                    "Unknown",
                    "N/A",
                    "N/A",
                    "N/A",
                    "N/A",
                    "N/A",
                    "Unknown",
                    "N/A",
                )

            
            # *** NEW *** Attempt to load your trained pipeline and predict
            try:
                PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
                models_dir = os.path.join(PROJECT_ROOT, "trained_models")
                chosen_model = selected_model  # from the store
                logging.debug(f"DEBUG: selected_model = {chosen_model}")
                model_path = os.path.join(models_dir, str(chosen_model))
                 
                best_model = joblib.load(model_path)  # or best_model.pkl
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
                    confidence_display = f"(信頼度スコア: {probability:.2f})"
                else:
                    model_prediction_text = "No valid features for model prediction"
            except Exception as e:
                logging.error(f"Error in model prediction: {e}")
                model_prediction_text = "Error in model prediction or Model not loaded"
                model_info_text = "Model not loaded"

        except Exception as e:
            error_msg = f"Error processing file: {str(e)}"
            logging.error(error_msg)
            return (
                normal_fig, 
                filtered_fig, 
                fft_fig,
                error_msg,
                True,
                
                False, 
                error_msg, 
                "danger",
                model_prediction_text, 
                analysis_summary, 
                
                model_prediction_text,
                {}, 
                no_update,
                
                no_update,
                no_update,
                no_update,
                no_update,
                no_update,
                no_update,
                no_update,
                no_update,
                no_update,
                no_update,
            )

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
            confidence_display,
            model_info_text,
        )
    
    @app.callback(
        Output("download-pdf", "data"),
        Input("download-pdf-btn", "n_clicks"),
        State("selected-file-1", "data"),
        prevent_initial_call=True
    )
    def download_pdf(n_clicks, file_path):
        if n_clicks:
            pdf_path = generate_pdf(file_path)
            if pdf_path:
                return dcc.send_file(pdf_path)
        return no_update
        
    @app.callback(
        Output("download-csv", "data"),
        Input("download-csv-btn", "n_clicks"),
        State("selected-file-1", "data"),
        prevent_initial_call=True
    )
    def download_csv(n_clicks, file_path):
        if n_clicks and file_path:
            logging.info(f"Download CSV button clicked with filepath='{file_path}'")
            return dcc.send_file(file_path)
        return no_update


def process_file(
    file_path,
    cutoff_freq,
    rms_window_size,
    hpf_rms_threshold,
    spike_threshold,
    y_axis_range,
    ball_size_id,
    operator_id,
):
    import pandas as pd
    from ..processing.data_loader import load_data
    import plotly.graph_objs as go
    import logging
    import os

    # graph layout settings
    layout = go.Layout(
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    
    normal_fig = go.Figure(

    )
    filtered_fig = go.Figure()
    fft_fig = go.Figure()
    features_dict = {}
    analysis_result_text = ""

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
        return [error_fig, go.Figure(), go.Figure()], {}, error_msg

    try:
        data = load_data(file_path)
        required_columns = ["X[mm]", "N[Ncm]"]
        for col in required_columns:
            if col not in data.columns:
                raise ValueError(f"Missing column {col}")

        x = data["X[mm]"]
        y = data["N[Ncm]"]
        
        # Extract features
        features_dict = extract_features(y.to_numpy(), fs=100.0, window_size=rms_window_size, threshold=7.5, kernel_size=1)
        
        # High-Pass Filtering
        y_filtered = high_pass_filter(y.to_numpy(), cutoff=cutoff_freq)
        filtered_series = pd.Series(y_filtered)
        filtered_rms = filtered_series.rolling(window=rms_window_size).apply(calculate_rms, raw=True)
        
        moving_max = filtered_series.rolling(window=100, min_periods=1).max()
        moving_min = filtered_series.rolling(window=100, min_periods=1).min()
        moving_max_avg = moving_max.rolling(window=int(rms_window_size), min_periods=1).mean()
        moving_min_avg = moving_min.rolling(window=int(rms_window_size), min_periods=1).mean()

        # Create figures
        normal_fig = create_normal_plot(x, y, y_axis_range)
        filtered_fig = create_filtered_plot(x, y_filtered, filtered_rms, moving_max_avg, moving_min_avg, cutoff_freq, [-1,1])

        
        from ..processing.feature_extraction import calculate_fft
        xf, amplitudes = calculate_fft(y.to_numpy(), fs=100.0)
        fft_fig = create_fft_plot(xf, amplitudes)

        # Analysis
        analysis_list = []
        from ..processing.anomaly_detection import detected_sudden_spike, analyse_hpf_rms
        spike_detected = detected_sudden_spike(filtered_rms, spike_threshold)
        if spike_detected:
            analysis_list.append("Sudden spike detected in filtered RMS data. Re-measurement recommended.")
        else:
            analysis_list.append("No sudden spike detected in filtered RMS data.")
        
        hpf_rms_result = analyse_hpf_rms(filtered_rms, hpf_rms_threshold)
        analysis_list.append(hpf_rms_result)
        
        if spike_detected or "over the threshold" in hpf_rms_result.lower():
            overall_result = "FAILED"
        else:
            overall_result = "PASSED"
        
        analysis_result_text = f"Overall analysis result: {overall_result}\n" + "\n".join(analysis_list)

        # Some extra meta
        import datetime
        operator_id = operator_id if operator_id else "Unknown Operator"
        timestamp_of_analysis = pd.Timestamp.now().isoformat()
        features_dict["Analysis Date"] = timestamp_of_analysis
        features_dict["Operator"] = operator_id
        features_dict["Ball Size"] = ball_size_id

        # Extract info from filename, e.g. 20241024151643_1505_XXX0001_20rpm.csv
        filename = os.path.basename(file_path)
        parts = filename.split("_")
        if len(parts) == 4:
            timestamp_of_measurement = parts[0]
            diameter_lead = parts[1]
            number = parts[2]
            rpm = parts[3].split(".")[0]
            features_dict["Measurement date"] = timestamp_of_measurement
            features_dict["Size"] = diameter_lead
            features_dict["Number"] = number
            features_dict["RPM"] = rpm

    except Exception as e:
        logging.error(f"Error processing file: {e}", exc_info=True)
        return [go.Figure(), go.Figure(), go.Figure()], {}, f"Error processing file: {e}"

    return ([normal_fig, filtered_fig, fft_fig], features_dict, analysis_result_text)
