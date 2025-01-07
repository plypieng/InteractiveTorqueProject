import dash
from dash import html, dcc, Input, Output, State, no_update
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
import plotly.graph_objs as go
import os
import pandas as pd
from flask import send_file, request, abort
import logging
from urllib.parse import quote_plus, unquote_plus
import uuid
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

from utils import (
    high_pass_filter,
    calculate_fft,
    calculate_rms,
    extract_all_features,
    load_data,
    is_safe_path,
    detected_sudden_spike,
    analyse_hpf_rms,
    SessionLocal,
    BallSize,
    Measurement,
    Feature,
    ALLOWED_DIRECTORY,
)
from plots import create_normal_plot, create_filtered_plot, create_fft_plot

# Configure logging
logging.basicConfig(
    level=logging.INFO, filename="app.log", format="%(asctime)s %(levelname)s:%(message)s"
)

# Initialize Dash app with Bootstrap theme and suppress callback exceptions
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.CYBORG],
    suppress_callback_exceptions=True,  # Allows callbacks for components not in the initial layout
)

# Load the trained model at startup
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'best_model.joblib')
if os.path.exists(MODEL_PATH):
    best_model = joblib.load(MODEL_PATH)
    logging.info("Trained model loaded successfully.")
else:
    best_model = None
    logging.warning("Trained model not found. Predictions will be disabled.")

# Help text for the user guide modal
help_text = """
### How to Use This Application
- **Select Measurement Files:** Choose up to two CSV files from the dropdowns to compare.
- **Select Ball Size:** Choose the appropriate ball size for the torque measurement.
- **Adjust Parameters:**
  - **High-Pass Filter Cutoff Frequency:** Enter a value in Hz.
  - **RMS Window Size:** Enter the window size for calculating moving RMS.
  - **HPF_RMS Threshold:** Set the threshold for HPF_RMS.
  - **Spike Threshold:** Set the threshold for spike detection in RMS data.
  - **Initial Torque Input:** Enter the initial torque measurement to receive a ball size recommendation.
  - **Y-axis Scale:** Use the slider to set the range.
- **View Graphs:** The plots will update based on your selections.
- **View Analysis Results:** PASS or FAILED results are displayed based on the analysis.
- **Download Data:** Use the links to download CSV or PDF files.
- **Label Data:** Assign labels to each dataset for model training.
"""

def get_tab1_content():
    return dbc.Container([
        html.H3("Step 1: Select Measurement Files and Ball Size", className="mb-4"),
        dbc.Row([
            dbc.Col([
                html.Label("Operator ID:"),
                dcc.Input(
                    id="operator-id-input",
                    type="text",
                    placeholder="Enter your ID",
                    className="mb-3",
                ),
                dbc.Tooltip(
                    "Provide your unique operator identifier.",
                    target="operator-id-input",
                    placement="right",
                ),
            ], width=6),
        ], className="mb-3"),

        dbc.Row([
            dbc.Col([
                html.Label("Ball Size:"),
                dcc.Dropdown(
                    id="ball-size-dropdown",
                    options=[],  # To be populated from the database
                    placeholder="Select a ball size",
                ),
                dbc.Tooltip(
                    "Select the appropriate ball size for torque measurement.",
                    target="ball-size-dropdown",
                    placement="right",
                ),
            ], width=6),
        ], className="mb-3"),

        dbc.Row([
            dbc.Col([
                html.Label("測定ファイル 1　選択:"),
                dcc.Dropdown(
                    id="file-dropdown-1",
                    options=[],
                    placeholder="Select a CSV file",
                ),
                dbc.Tooltip(
                    "Select the primary CSV file containing torque measurement data.",
                    target="file-dropdown-1",
                    placement="right",
                ),
                dbc.Button(
                    "Proceed to Visualization",
                    id="proceed-visualization-btn",
                    n_clicks=0,
                    color="primary",
                    className="mt-3",
                    disabled=True,  # Initially disabled
                ),
            ], width=6),
        ], className="mb-3"),

        dbc.Row([
            dbc.Col([
                html.Label("測定ファイル 2　選択 (任意):"),
                dcc.Dropdown(
                    id="file-dropdown-2",
                    options=[],
                    placeholder="Select a CSV file",
                ),
                dbc.Tooltip(
                    "Optionally select a second CSV file for comparison.",
                    target="file-dropdown-2",
                    placement="right",
                ),
            ], width=6),
        ], className="mb-3"),
    ])

# Data Visualization tab content
def get_tab2_content():
    return dbc.Container([
        html.H3("Step 2: Data Visualization", className="mb-4"),
        
        # Anomaly Alert
        dbc.Alert(id="anomaly-alert", is_open=False, duration=4000),
        
        # Parameter Inputs
        dbc.Card([
            dbc.CardHeader("Adjust Parameters"),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        html.Label("High-Pass Filter Cutoff Frequency (Hz):"),
                        dbc.Input(
                            id="cutoff-input",
                            type="number",
                            min=0.01,
                            step=0.01,
                            placeholder="Enter cutoff frequency",
                            value=10,  # Default value
                        ),
                        dbc.Tooltip(
                            "Set the cutoff frequency for the high-pass filter.",
                            target="cutoff-input",
                            placement="right",
                        ),
                    ], md=4),
                    dbc.Col([
                        html.Label("RMS Window Size:"),
                        dbc.Input(
                            id="rms-window-size",
                            type="number",
                            min=1,
                            step=1,
                            placeholder="Enter window size",
                            value=300,  # Default value
                        ),
                        dbc.Tooltip(
                            "Set the window size for calculating moving RMS.",
                            target="rms-window-size",
                            placement="right",
                        ),
                    ], md=4),
                    dbc.Col([
                        html.Label("HPF_RMS Threshold:"),
                        dbc.Input(
                            id="hpf-rms-threshold",
                            type="number",
                            min=0.005,
                            step=0.005,
                            placeholder="Enter HPF_RMS threshold",
                            value=0.05,  # Default value
                        ),
                        dbc.Tooltip(
                            "Set the threshold for HPF_RMS analysis.",
                            target="hpf-rms-threshold",
                            placement="right",
                        ),
                    ], md=4),
                ]),
                dbc.Row([
                    dbc.Col([
                        html.Label("Spike Threshold:"),
                        dbc.Input(
                            id="spike-threshold",
                            type="number",
                            min=0.0,
                            step=0.01,
                            placeholder="Enter spike threshold",
                            value=0.1,  # Default value
                        ),
                        dbc.Tooltip(
                            "Set the threshold for detecting sudden spikes in RMS data.",
                            target="spike-threshold",
                            placement="right",
                        ),
                    ], md=4),
                    dbc.Col([
                        html.Label("Initial Torque Input (Nm):"),
                        dbc.Input(
                            id="initial-torque-input",
                            type="number",
                            min=0.0,
                            step=0.01,
                            placeholder="Enter initial torque",
                            value=0.0,  # Default value
                        ),
                        dbc.Tooltip(
                            "Enter the initial torque measurement to receive a ball size recommendation.",
                            target="initial-torque-input",
                            placement="right",
                        ),
                    ], md=4),
                    dbc.Col([
                        html.Label("Y-axis Scale:"),
                        dcc.RangeSlider(
                            id="y-axis-slider",
                            min=0,
                            max=20,
                            step=0.5,
                            value=[0, 10],  # Default range
                            marks={i: str(i) for i in range(0, 21, 5)},
                            tooltip={"placement": "bottom", "always_visible": True},
                        ),
                        dbc.Tooltip(
                            "Adjust the Y-axis scale for the plots.",
                            target="y-axis-slider",
                            placement="right",
                        ),
                    ], md=4),
                ], className="mt-4"),
            ])
        ], className="mb-4"),
        
        # Model Prediction Display
        dbc.Row([
            dbc.Col([
                html.Div(id="model-prediction", className="text-success", style={"fontWeight": "bold", "fontSize": "1.2em"}),
            ], width=12),
        ], className="mb-3"),
        
        # Graphs
        dbc.Row([
            dbc.Col([
                dcc.Loading(
                    id="loading-normal-graph",
                    type="default",
                    children=dcc.Graph(id="normal-graph"),
                )
            ], width=6),
            dbc.Col([
                dcc.Loading(
                    id="loading-filtered-graph",
                    type="default",
                    children=dcc.Graph(id="filtered-graph"),
                )
            ], width=6),
        ], className="mb-3"),
        dbc.Row([
            dbc.Col([
                dcc.Loading(
                    id="loading-fft-graph",
                    type="default",
                    children=dcc.Graph(id="fft-graph"),
                )
            ], width=6),
            dbc.Col([
                html.H5("特徴量"),
                html.Div(id="features", style={"whiteSpace": "pre-wrap"}),
                html.H5("分析結果"),
                html.Div(
                    id="analysis-result",
                    style={"whiteSpace": "pre-wrap", "fontWeight": "bold", "fontSize": "1.2em"},
                ),
            ], width=6),
        ], className="mb-3"),

        dbc.Button(
            "Back to File Selection",
            id="back-to-file-selection-btn",
            n_clicks=0,
            color="secondary",
            className="mt-3",
        ),
        dbc.Button(
            "Proceed to Labeling",
            id="proceed-labeling-btn",
            n_clicks=0,
            color="info",
            className="mt-3 ms-2",
            disabled=True,  # Initially disabled
        ),
    ])

def get_tab3_content():
    return dbc.Container([
        html.H3("Step 3: Data Labeling", className="mb-4"),
        dbc.Row([
            dbc.Col([
                html.Label("Select Label:"),
                dcc.Dropdown(
                    id="data-label-dropdown",
                    options=[
                        {"label": "Pass", "value": "PASS"},
                        {"label": "Fail", "value": "FAIL"},
                        {"label": "Requires Inspection", "value": "REQUIRES_INSPECTION"},
                    ],
                    placeholder="Select a label",
                ),
                dbc.Tooltip(
                    "Assign a label to the selected dataset to indicate its quality.",
                    target="data-label-dropdown",
                    placement="right",
                ),
            ], width=6),
        ], className="mb-3"),

        dbc.Button(
            "Back to Visualization",
            id="back-to-visualization-btn",
            n_clicks=0,
            color="secondary",
            className="mt-3",
        ),
        dbc.Button(
            "Save Label and Proceed",
            id="save-label-btn",
            n_clicks=0,
            color="primary",
            className="mt-3 ms-2",
            disabled=True,  # Initially disabled
        ),
        
        # Alert for Save Label
        html.Div(id="save-alert-message", className="mt-3"),
    ])

def get_tab4_content():
    return dbc.Container([
        html.H3("Step 4: Review & Submit", className="mb-4"),
        dbc.Row([
            dbc.Col([
                html.H5("Selected Files:"),
                html.Ul(id="review-selected-files"),
            ], width=6),
        ], className="mb-3"),

        dbc.Row([
            dbc.Col([
                html.H5("Assigned Labels:"),
                html.Ul(id="review-assigned-labels"),
            ], width=6),
        ], className="mb-3"),

        dbc.Button(
            "Back to Labeling",
            id="back-to-labeling-btn",
            n_clicks=0,
            color="secondary",
            className="mt-3",
        ),
        dbc.Button(
            "Submit Labels",
            id="submit-labels-btn",
            n_clicks=0,
            color="success",
            className="mt-3 ms-2",
            disabled=True,  # Initially disabled
        ),
        
        # Alert for Submit Labels
        html.Div(id="submit-alert-message", className="mt-3"),
    ])

# App Layout with initial tab content loaded
app.layout = dbc.Container([
    html.H1("トルクデーター解析 V1.1", className="text-center text-primary mb-4"),

    # Help button and modal
    dbc.Button("Help", id="open-modal", n_clicks=0, className="mb-3"),
    dbc.Modal(
        [
            dbc.ModalHeader("User Guide"),
            dbc.ModalBody(dcc.Markdown(help_text)),
            dbc.ModalFooter(
                dbc.Button("Close", id="close-modal", className="ml-auto")
            ),
        ],
        id="modal",
        is_open=False,
    ),

    # Interval component for updating file and ball size options
    dcc.Interval(
        id="interval-component",
        interval=10*1000,  # Update every 10 seconds
        n_intervals=0,
    ),

    # Hidden stores to keep state
    dcc.Store(id="selected-file-1"),
    dcc.Store(id="selected-file-2"),
    dcc.Store(id="selected-ball-size"),
    dcc.Store(id="labels-data", data={}),  # To store labels
    dcc.Store(id="prev-file-list", data=[]),  # To track previous file list

    # Progress bar
    dbc.Progress(id="progress-bar", value=25, striped=True, animated=True, className="mb-4"),

    # Step-by-step Tabs with unique IDs for each tab
    dbc.Tabs([
        dbc.Tab(label="File Selection", tab_id="tab-1", id="tab-1", children=get_tab1_content()),
        dbc.Tab(label="Data Visualization", tab_id="tab-2", id="tab-2", children=get_tab2_content(), disabled=True),
        dbc.Tab(label="Data Labeling", tab_id="tab-3", id="tab-3", children=get_tab3_content(), disabled=True),
        dbc.Tab(label="Review & Submit", tab_id="tab-4", id="tab-4", children=get_tab4_content(), disabled=True),
    ], id="tabs", active_tab="tab-1"),
], fluid=True)

# Callback to open and close the help modal
@app.callback(
    Output("modal", "is_open"),
    [Input("open-modal", "n_clicks"), Input("close-modal", "n_clicks")],
    [State("modal", "is_open")],
)
def toggle_modal(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open

# Callback to update file options only when the file list changes
@app.callback(
    [
        Output("file-dropdown-1", "options"),
        Output("file-dropdown-1", "value"),
        Output("file-dropdown-2", "options"),
        Output("file-dropdown-2", "value"),
        Output("prev-file-list", "data"),
    ],
    [
        Input("interval-component", "n_intervals"),
    ],
    [
        State("file-dropdown-1", "value"),
        State("file-dropdown-2", "value"),
        State("prev-file-list", "data"),
    ],
)
def update_file_options(n_intervals, selected_file1, selected_file2, prev_file_list):
    directory = ALLOWED_DIRECTORY
    try:
        if not os.path.exists(directory):
            logging.warning(f"Allowed directory does not exist: {directory}")
            current_files = []
        else:
            current_files = [f for f in os.listdir(directory) if f.endswith(".csv")]
        # Sort files by modification time, newest first
        files_sorted = sorted(
            current_files, key=lambda x: os.path.getmtime(os.path.join(directory, x)), reverse=True
        )
        options_sorted = [{"label": f, "value": os.path.join(directory, f)} for f in files_sorted]
        
        # Check if the file list has changed
        if sorted(current_files) != sorted(prev_file_list):
            valid_values = [option["value"] for option in options_sorted]
            value1 = selected_file1 if selected_file1 in valid_values else None
            value2 = selected_file2 if selected_file2 in valid_values else None
            return options_sorted, value1, options_sorted, value2, current_files
        else:
            return no_update, no_update, no_update, no_update, no_update
    except Exception as e:
        logging.error(f"Error accessing directory {directory}: {e}")
        return no_update, no_update, no_update, no_update, no_update

# Callback to populate ball size options from the database
@app.callback(
    Output("ball-size-dropdown", "options"),
    Input("interval-component", "n_intervals")
)
def update_ball_size_options(n_intervals):
    session = SessionLocal()
    try:
        ball_sizes = session.query(BallSize).all()
        options = [{"label": bs.size, "value": bs.id} for bs in ball_sizes]
        return options
    except Exception as e:
        logging.error(f"Error fetching ball sizes: {e}")
        return []
    finally:
        session.close()

# Callback to enable proceed button based on selections
@app.callback(
    Output("proceed-visualization-btn", "disabled"),
    [
        Input("file-dropdown-1", "value"),
        Input("ball-size-dropdown", "value"),
        Input("operator-id-input", "value"),
    ],
)
def enable_proceed_button(file1, ball_size_id, operator_id):
    if file1 and ball_size_id and operator_id:
        return False
    return True

# Callback to store selected files and ball size
@app.callback(
    [
        Output("selected-file-1", "data"),
        Output("selected-file-2", "data"),
        Output("selected-ball-size", "data"),
    ],
    [
        Input("proceed-visualization-btn", "n_clicks"),
    ],
    [
        State("file-dropdown-1", "value"),
        State("file-dropdown-2", "value"),
        State("ball-size-dropdown", "value"),
    ],
    prevent_initial_call=True,
)
def store_selected_files(proceed_clicks, file1, file2, ball_size_id):
    if proceed_clicks:
        return file1, file2, ball_size_id
    return no_update, no_update, no_update

# Callback to update progress bar
@app.callback(
    Output("progress-bar", "value"),
    [
        Input("tabs", "active_tab"),
    ],
)
def update_progress(active_tab):
    progress = {
        "tab-1": 25,
        "tab-2": 50,
        "tab-3": 75,
        "tab-4": 100,
    }
    return progress.get(active_tab, 25)

@app.callback(
    Output("progress-bar", "label"),
    [
        Input("tabs", "active_tab"),
    ],
)
def update_progress_label(active_tab):
    labels = {
        "tab-1": "Step 1 of 4: File Selection",
        "tab-2": "Step 2 of 4: Data Visualization",
        "tab-3": "Step 3 of 4: Data Labeling",
        "tab-4": "Step 4 of 4: Review & Submit",
    }
    return labels.get(active_tab, "Step 1 of 4: File Selection")

# Callback to enable Data Visualization tab
@app.callback(
    Output("tab-2", "disabled"),
    [
        Input("proceed-visualization-btn", "n_clicks"),
    ],
    [
        State("tab-2", "disabled"),
    ],
    prevent_initial_call=True,
)
def enable_data_visualization(proceed_vis_clicks, current_disabled):
    if proceed_vis_clicks and current_disabled:
        return False
    return current_disabled

# Callback to enable Data Labeling tab
@app.callback(
    Output("tab-3", "disabled"),
    [
        Input("proceed-labeling-btn", "n_clicks"),
    ],
    [
        State("tab-3", "disabled"),
    ],
    prevent_initial_call=True,
)
def enable_data_labeling(proceed_lab_clicks, current_disabled):
    if proceed_lab_clicks and current_disabled:
        return False
    return current_disabled

# Callback to enable Review & Submit tab
@app.callback(
    Output("tab-4", "disabled"),
    [
        Input("submit-labels-btn", "n_clicks"),
        Input("save-label-btn", "n_clicks"),
    ],
    [
        State("tab-4", "disabled"),
    ],
    prevent_initial_call=True,
)
def enable_review_submit(submit_clicks, save_label_clicks, current_disabled):
    
    if submit_clicks and current_disabled:
        return False
    return current_disabled

# Callback to navigate between tabs
@app.callback(
    Output("tabs", "active_tab"),
    [
        Input("proceed-visualization-btn", "n_clicks"),
        Input("back-to-file-selection-btn", "n_clicks"),
        Input("proceed-labeling-btn", "n_clicks"),
        Input("back-to-visualization-btn", "n_clicks"),
        Input("submit-labels-btn", "n_clicks"),
        Input("back-to-labeling-btn", "n_clicks"),
    ],
    [
        State("tabs", "active_tab"),
    ],
    prevent_initial_call=True,
)
def navigate_tabs(
    proceed_vis_clicks,
    back_to_file_clicks,
    proceed_lab_clicks,
    back_to_vis_clicks,
    submit_labels_clicks,
    back_to_labeling_clicks,
    current_tab,
):
    ctx = dash.callback_context

    if not ctx.triggered:
        raise PreventUpdate

    button_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if button_id == "proceed-visualization-btn" and current_tab == "tab-1":
        return "tab-2"
    elif button_id == "back-to-file-selection-btn" and current_tab == "tab-2":
        return "tab-1"
    elif button_id == "proceed-labeling-btn" and current_tab == "tab-2":
        return "tab-3"
    elif button_id == "back-to-visualization-btn" and current_tab == "tab-3":
        return "tab-2"
    elif button_id == "submit-labels-btn" and current_tab == "tab-3":
        return "tab-4"
    elif button_id == "back-to-labeling-btn" and current_tab == "tab-4":
        return "tab-3"

    return no_update

# Callback to update graphs and perform feature extraction with model inference
@app.callback(
    [
        Output("normal-graph", "figure"),
        Output("filtered-graph", "figure"),
        Output("fft-graph", "figure"),
        Output("features", "children"),
        Output("analysis-result", "children"),
        Output("proceed-labeling-btn", "disabled"),
        Output("anomaly-alert", "is_open"),
        Output("anomaly-alert", "children"),
        Output("anomaly-alert", "color"),
        Output("model-prediction", "children"),  # New output for model prediction
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
)
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
    normal_fig = go.Figure()
    filtered_fig = go.Figure()
    fft_fig = go.Figure()
    features_text = ""
    analysis_result_text = ""
    proceed_disabled = True  
    anomaly_is_open = False
    anomaly_message = ""
    anomaly_color = "success"
    model_prediction_text = ""
    
    if not (file_path_1 and cutoff_freq and rms_window_size and hpf_rms_threshold and spike_threshold and y_axis_range and ball_size_id and operator_id):
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
        )
    
    session = SessionLocal()
    try:
        # Load data
        data = load_data(file_path_1)
        torque_values = data["N[Ncm]"].to_numpy()
        
        # Perform anomaly detection
        anomaly, message = detect_anomalous_measurement(torque_values, ball_size_id, session)
        if anomaly:
            analysis_result_text = f"ANOMALY DETECTED: {message}"
            proceed_disabled = True
            anomaly_is_open = True
            anomaly_message = message
            anomaly_color = "danger"
        else:
            # Proceed with feature extraction and plotting
            figs, features_text, analysis_result_text = process_file(
                file_path_1,
                cutoff_freq,
                rms_window_size,
                hpf_rms_threshold,
                spike_threshold,
                y_axis_range,
                "1",
                ball_size_id,
                operator_id
            )
            if figs:
                normal_fig, filtered_fig, fft_fig = figs
                # Determine if proceeding to labeling should be enabled based on analysis
                if "No sudden spike detected in filtered RMS data" in analysis_result_text:
                    proceed_disabled = False
                else:
                    proceed_disabled = True
                
                # Perform model inference if model is loaded
                if best_model:
                    # Fetch features from the database
                    measurement = session.query(Measurement).filter(Measurement.file_path == file_path_1).first()
                    if measurement:
                        feature_dict = {feature.feature_name: feature.feature_value for feature in measurement.features}
                        # Exclude non-feature fields if any
                        if 'Label' in feature_dict:
                            del feature_dict['Label']
                        feature_df = pd.DataFrame([feature_dict])
                        
                        # Handle scaling if necessary (assuming scaler was used during training)
                        scaler = StandardScaler()
                        # Load scaler if it was saved; placeholder here
                        # scaler = joblib.load('models/scaler.joblib')
                        # For simplicity, assuming features are already scaled or scaler is part of the pipeline
                        prediction = best_model.predict(feature_df)[0]
                        if hasattr(best_model, 'predict_proba'):
                            probability = best_model.predict_proba(feature_df)[0][1]
                        else:
                            probability = 0.0  # Assign default if predict_proba not available
                        
                        prediction_label = "PASS" if prediction == 1 else "FAIL"
                        model_prediction_text = f"Model Prediction: {prediction_label} (Confidence: {probability:.2f})"
                    else:
                        model_prediction_text = "Measurement not found for prediction."
                else:
                    model_prediction_text = "Model not loaded."
    except Exception as e:
        logging.error(f"Error updating graphs: {e}")
    finally:
        session.close()
    
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
    )

def detect_anomalous_measurement(torque_values, ball_size_id, session):
    """
    Detect anomalies based on ball size and torque measurements.
    """
    try:
        ball_size = session.query(BallSize).filter(BallSize.id == ball_size_id).first()
        if not ball_size:
            return True, "Unknown ball size."
        
        # Define acceptable torque range
        torque_min = ball_size.torque_min
        torque_max = ball_size.torque_max
        
        # Calculate summary statistics
        torque_mean = np.mean(torque_values)
        torque_std = np.std(torque_values)
        
        # Check if mean torque is within acceptable range
        if not (torque_min <= torque_mean <= torque_max):
            return True, f"Mean torque {torque_mean:.2f} Nm out of range ({torque_min}-{torque_max} Nm)."
        
        # Additional checks (e.g., waviness magnitude) can be added here
        
        return False, "No anomalies detected."
    except Exception as e:
        logging.error(f"Error in anomaly detection: {e}")
        return True, "Error during anomaly detection."

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
    # Security check
    if not is_safe_path(ALLOWED_DIRECTORY, file_path):
        logging.warning(f"Attempt to access invalid file path: {file_path}")
        error_fig = go.Figure()
        error_fig.add_annotation(
            text="Invalid file path selected.",
            xref="paper",
            yref="paper",
            showarrow=False,
            font=dict(color="red", size=16),
        )
        return [error_fig, go.Figure(), go.Figure()], "", ""
    
    try:
        # Read data
        data = load_data(file_path)
        x = data["X[mm]"]
        y = data["N[Ncm]"]
        
        # High-Pass Filtering
        y_filtered = high_pass_filter(y.to_numpy(), cutoff=cutoff_freq)
        filtered_series = pd.Series(y_filtered)
        filtered_rms = filtered_series.rolling(window=rms_window_size).apply(calculate_rms, raw=True)
        
        # Compute moving max and min of the filtered data
        moving_max = filtered_series.rolling(window=100, min_periods=1).max()
        moving_min = filtered_series.rolling(window=100, min_periods=1).min()
        
        # Compute moving average of the moving max and min
        moving_max_avg = moving_max.rolling(window=int(rms_window_size), min_periods=1).mean()
        moving_min_avg = moving_min.rolling(window=int(rms_window_size), min_periods=1).mean()
        
        # FFT
        xf, amplitudes = calculate_fft(y.to_numpy())
        
        # Create plots
        normal_fig = create_normal_plot(x, y, y_axis_range)
        filtered_fig = create_filtered_plot(
            x, y_filtered, filtered_rms, moving_max_avg, moving_min_avg, cutoff_freq, y_axis_range
        )
        fft_fig = create_fft_plot(xf, amplitudes)
        
        # Extract features
        features = extract_all_features(y, cutoff=cutoff_freq)
        features_text = "\n".join([f"{key}: {value}" for key, value in features.items()])
        
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
        
        operator_id = operator_id if operator_id else "Unknown Operator"
        timestamp = pd.Timestamp.now().isoformat()
        features_text += f"\nOperator ID: {operator_id}\nTimestamp: {timestamp}"
        
        # Store measurement and features in the database
        session = SessionLocal()
        try:
            measurement = Measurement(
                file_path=file_path,
                operator_id=operator_id,
                ball_size_id=ball_size_id,
                timestamp=pd.Timestamp.now(),
                status='Processed'
            )
            session.add(measurement)
            session.commit()
            
            # Add features
            for feature_name, feature_value in features.items():
                feature = Feature(
                    measurement_id=measurement.id,
                    feature_name=feature_name,
                    feature_value=feature_value
                )
                session.add(feature)
            session.commit()
        except Exception as e:
            logging.error(f"Error storing measurement and features: {e}")
            session.rollback()
        finally:
            session.close()
        
        return [normal_fig, filtered_fig, fft_fig], features_text, analysis_result_text
    except Exception as e:
        logging.error(f"Error processing file {file_path}: {e}")
        error_fig = go.Figure()
        error_fig.add_annotation(
            text=f"Error processing data: {e}",
            xref="paper",
            yref="paper",
            showarrow=False,
            font=dict(color="red", size=16),
        )
        return [error_fig, go.Figure(), go.Figure()], "", ""

if __name__ == "__main__":
    app.run_server(debug=True, host="127.0.0.1", port=8050)