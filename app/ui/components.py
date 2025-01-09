# app/ui/components.py
from dash import html, dcc
import dash_bootstrap_components as dbc

def get_loading_overlay():
    return dbc.Spinner(
        html.Div(id="loading-overlay"),
        fullscreen=True,
        color="primary",
        type="grow",
    )

def get_confirmation_modal():
    return dbc.Modal([
        dbc.ModalHeader(dbc.ModalTitle([
            html.I(className="fas fa-check-circle me-2"),
            "Confirm Submission"
        ])),
        dbc.ModalBody([
            html.P("Are you sure you want to submit the following data?"),
            html.Div(id="confirmation-content", className="mt-3"),
        ]),
        dbc.ModalFooter([
            dbc.Button(
                "Cancel",
                id="cancel-submission",
                color="secondary",
                className="me-2"
            ),
            dbc.Button(
                "Confirm Submission",
                id="confirm-submission",
                color="success"
            ),
        ]),
    ], id="confirmation-modal", is_open=False)

def get_progress_indicator():
    return html.Div([
        dbc.Progress(
            value=25,
            id="progress-bar",
            striped=True,
            animated=True,
            className="mb-3",
            style={"height": "8px"}
        ),
        html.Div(
            id="progress-label",
            className="text-center text-muted small mb-4",
        ),
    ])

def get_help_modal():
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
    
    return dbc.Modal([
        dbc.ModalHeader(dbc.ModalTitle([
            html.I(className="fas fa-book me-2"),
            "User Guide"
        ])),
        dbc.ModalBody(dcc.Markdown(help_text)),
        dbc.ModalFooter(
            dbc.Button("Close", id="close-modal", className="ms-auto")
        ),
    ], id="modal", is_open=False, size="lg")

def get_tab1_content():
    return dbc.Container([
        html.H3("Step 1: Select Measurement Files and Ball Size", className="mb-4"),
        dbc.Row([
            dbc.Col([
                html.Label("Operator ID:", className="fw-bold"),
                dbc.Input(
                    id="operator-id-input",
                    type="text",
                    placeholder="Enter your ID",
                    className="mb-2",
                    valid=False,  # Initialize as not valid
                    invalid=False,  # Initialize as not invalid
                ),
                dbc.FormFeedback(
                    "Please enter your operator ID",
                    id="operator-id-feedback",
                ),
                dbc.Tooltip(
                    "Provide your unique operator identifier",
                    target="operator-id-input",
                    placement="right",
                    delay={"show": 200, "hide": 0},
                ),
            ], width=12, md=6),
        ], className="mb-4"),

        dbc.Row([
            dbc.Col([
                html.Label("Ball Size:", className="fw-bold"),
                dbc.Select(
                    id="ball-size-dropdown",
                    options=[],
                    placeholder="Select a ball size",
                    className="mb-2",
                    value=None,
                ),
                dbc.FormFeedback(
                    "Please select a ball size",
                    id="ball-size-feedback",
                ),
                dbc.Tooltip(
                    "Select the appropriate ball size for torque measurement",
                    target="ball-size-dropdown",
                    placement="right",
                    delay={"show": 200, "hide": 0},
                ),
            ], width=12, md=6),
        ], className="mb-4"),

        dbc.Row([
            dbc.Col([
                html.Label("測定ファイル 1　選択:", className="fw-bold"),
                dbc.Select(
                    id="file-dropdown-1",
                    options=[],
                    placeholder="Select a CSV file",
                    className="mb-2",
                    value=None,
                ),
                dbc.FormFeedback(
                    "Please select a primary CSV file",
                    id="file-1-feedback",
                ),
                dbc.Tooltip(
                    "Select the primary measurement CSV file",
                    target="file-dropdown-1",
                    placement="right",
                    delay={"show": 200, "hide": 0},
                ),
            ], width=12, md=6),
        ], className="mb-4"),

        dbc.Row([
            dbc.Col([
                html.Label("測定ファイル 2　選択 (任意):", className="fw-bold"),
                dbc.Select(
                    id="file-dropdown-2",
                    options=[],
                    placeholder="Select a CSV file (optional)",
                    className="mb-2",
                    value=None,
                ),
                dbc.Tooltip(
                    "Select an additional measurement CSV file for comparison (optional)",
                    target="file-dropdown-2",
                    placement="right",
                    delay={"show": 200, "hide": 0},
                ),
            ], width=12, md=6),
        ], className="mb-4"),

        dbc.Button(
            [
                html.I(className="fas fa-arrow-right me-2"),
                "Proceed to Visualization"
            ],
            id="proceed-visualization-btn",
            n_clicks=0,
            color="primary",
            className="mt-3",
            disabled=True,
            size="lg",
        ),
    ])

def get_tab2_content():
    return dbc.Container([
        html.H3("Step 2: Data Visualization", className="mb-4"),
        
        # Model Prediction Alert
        dbc.Alert(
            id="model-prediction",
            className="mb-4",
            dismissable=True,
            duration=4000,
            style={"fontSize": "1.1em"}
        ),
        
        # Anomaly Alert with icon
        dbc.Alert(
            [
                html.I(className="fas fa-exclamation-triangle me-2"),
                html.Span(id="anomaly-alert-text")
            ],
            id="anomaly-alert",
            is_open=False,
            duration=4000,
            className="mb-4"
        ),
        
        # Parameter Inputs Card
        dbc.Card([
            dbc.CardHeader([
                html.I(className="fas fa-sliders-h me-2"),
                "Analysis Parameters"
            ], className="fw-bold"),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        html.Label("High-Pass Filter Cutoff Frequency (Hz):", className="fw-bold"),
                        dbc.InputGroup([
                            dbc.Input(
                                id="cutoff-input",
                                type="number",
                                min=0.01,
                                step=0.01,
                                value=10,
                                className="mb-2",
                            ),
                            dbc.InputGroupText("Hz"),
                        ]),
                        dbc.FormText("Recommended range: 0.01 - 50 Hz"),
                    ], md=4),
                    dbc.Col([
                        html.Label("RMS Window Size:", className="fw-bold"),
                        dbc.Input(
                            id="rms-window-size",
                            type="number",
                            min=1,
                            step=1,
                            value=300,
                            className="mb-2",
                        ),
                        dbc.FormText("Recommended range: 100 - 500"),
                    ], md=4),
                    dbc.Col([
                        html.Label("HPF_RMS Threshold:", className="fw-bold"),
                        dbc.Input(
                            id="hpf-rms-threshold",
                            type="number",
                            min=0.005,
                            step=0.005,
                            value=0.05,
                            className="mb-2",
                        ),
                        dbc.FormText("Recommended range: 0.005 - 0.1"),
                    ], md=4),
                ], className="mb-3"),
                dbc.Row([
                    dbc.Col([
                        html.Label("Spike Threshold:", className="fw-bold"),
                        dbc.Input(
                            id="spike-threshold",
                            type="number",
                            min=0.0,
                            step=0.01,
                            value=0.1,
                            className="mb-2",
                        ),
                        dbc.FormText("Recommended range: 0.05 - 0.2"),
                    ], md=4),
                    dbc.Col([
                        html.Label("Initial Torque (Nm):", className="fw-bold"),
                        dbc.InputGroup([
                            dbc.Input(
                                id="initial-torque-input",
                                type="number",
                                min=0.0,
                                step=0.01,
                                value=0.0,
                                className="mb-2",
                            ),
                            dbc.InputGroupText("Nm"),
                        ]),
                    ], md=4),
                    dbc.Col([
                        html.Label("Y-axis Scale:", className="fw-bold"),
                        dcc.RangeSlider(
                            id="y-axis-slider",
                            min=0,
                            max=20,
                            step=0.5,
                            value=[0, 10],
                            marks={i: str(i) for i in range(0, 21, 5)},
                            tooltip={"placement": "bottom", "always_visible": True},
                            className="mb-2",
                        ),
                    ], md=4),
                ]),
            ]),
        ], className="mb-4"),
        
        # Graphs Section
        html.Div([
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader([
                            html.I(className="fas fa-chart-line me-2"),
                            "Raw Data"
                        ], className="fw-bold"),
                        dbc.CardBody(
                            dcc.Loading(
                                id="loading-normal-graph",
                                type="circle",
                                children=dcc.Graph(id="normal-graph"),
                            )
                        ),
                    ]),
                ], md=6),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader([
                            html.I(className="fas fa-filter me-2"),
                            "Filtered Data"
                        ], className="fw-bold"),
                        dbc.CardBody(
                            dcc.Loading(
                                id="loading-filtered-graph",
                                type="circle",
                                children=dcc.Graph(id="filtered-graph"),
                            )
                        ),
                    ]),
                ], md=6),
            ], className="mb-4"),
            
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader([
                            html.I(className="fas fa-wave-square me-2"),
                            "FFT Analysis"
                        ], className="fw-bold"),
                        dbc.CardBody(
                            dcc.Loading(
                                id="loading-fft-graph",
                                type="circle",
                                children=dcc.Graph(id="fft-graph"),
                            )
                        ),
                    ]),
                ], md=6),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader([
                            html.I(className="fas fa-list-alt me-2"),
                            "Analysis Results"
                        ], className="fw-bold"),
                        dbc.CardBody([
                            html.H5([
                                html.I(className="fas fa-chart-bar me-2"),
                                "特徴量"
                            ]),
                            html.Div(
                                id="features",
                                style={"whiteSpace": "pre-wrap"},
                                className="mb-4"
                            ),
                            html.H5([
                                html.I(className="fas fa-clipboard-check me-2"),
                                "分析結果"
                            ]),
                            html.Div(
                                id="analysis-result",
                                style={
                                    "whiteSpace": "pre-wrap",
                                    "fontWeight": "bold",
                                    "fontSize": "1.2em"
                                }
                            ),
                        ]),
                    ]),
                ], md=6),
            ]),
        ], className="mb-4"),
        
        # Navigation Buttons
        dbc.Row([
            dbc.Col([
                dbc.Button(
                    [
                        html.I(className="fas fa-arrow-left me-2"),
                        " Back to File Selection"
                    ],
                    id="back-to-file-selection-btn",
                    color="secondary",
                    className="me-2",
                ),
                dbc.Button(
                    [
                        html.I(className="fas fa-file-csv me-2"),
                        " Download CSV",
                        dcc.Download(id="download-csv"),
                    ],
                    id="download-csv-btn",
                    color="primary",
                ),
                dbc.Button(
                    [
                        html.I(className="fas fa-file-pdf ms-2"),
                        " Download PDF",         
                        dcc.Download(id="download-pdf"),
                    ],
                    id="download-pdf-btn",
                    color="primary",
                ),
                dbc.Button(
                    [
                        html.I(className="fas fa-arrow-right me-2"),
                        " Proceed to Labeling"
                    ],
                    id="proceed-labeling-btn",
                    color="success",
                    disabled=True,
                ),
            ], className="d-flex justify-content-between"),
        ]),
    ])

def get_tab3_content():
    return dbc.Container([
        html.H3("Step 3: Data Labeling", className="mb-4"),
        
        # File Summary Card
        dbc.Card([
            dbc.CardHeader([
                html.I(className="fas fa-file-alt me-2"),
                "Selected File Summary"
            ], className="fw-bold"),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        html.P([
                            html.Strong("File: "),
                            html.Span(id="selected-file-name", style={"whiteSpace": "pre-wrap"})
                        ]),
                        html.P([
                            html.Strong("Ball Size: "),
                            html.Span(id="selected-ball-size-name")
                        ]),
                        html.P([
                            html.Strong("Operator: "),
                            html.Span(id="selected-operator-name")
                        ]),
                    ], md=6),
                    dbc.Col([
                        html.P([
                            html.Strong("Analysis Result: "),
                            html.Span(id="analysis-summary", className="fw-bold")
                        ]),
                        html.P([
                            html.Strong("Model Prediction: "),
                            html.Span(id="model-prediction-summary", className="fw-bold")
                        ]),
                    ], md=6),
                ]),
            ]),
        ], className="mb-4"),
        
        # Labeling Card
        dbc.Card([
            dbc.CardHeader([
                html.I(className="fas fa-tags me-2"),
                "Assign Label"
            ], className="fw-bold"),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        html.Label("Select Label:", className="fw-bold mb-2"),
                        dbc.RadioItems(
                            id="data-label-dropdown",
                            options=[
                                {"label": "Pass", "value": "PASS"},
                                {"label": "Fail", "value": "FAIL"},
                                {"label": "Requires Inspection", "value": "REQUIRES_INSPECTION"},
                            ],
                            inline=True,
                            className="mb-3",
                        ),
                        dbc.FormText("Choose the appropriate label based on your analysis"),
                    ], width=12),
                ]),
                dbc.Row([
                    dbc.Col([
                        html.Label("Additional Notes:", className="fw-bold mb-2"),
                        dbc.Textarea(
                            id="label-notes",
                            placeholder="Enter any additional observations or notes...",
                            style={"height": "100px"},
                            className="mb-3",
                        ),
                    ], width=12),
                ]),
            ]),
        ], className="mb-4"),
        
        # Alert for feedback
        dbc.Alert(
            id="save-alert-message",
            dismissable=True,
            duration=4000,
            is_open=False,
        ),
        
        # Navigation Buttons
        dbc.Row([
            dbc.Col([
                dbc.Button(
                    [
                        html.I(className="fas fa-arrow-left me-2"),
                        "Back to Visualization"
                    ],
                    id="back-to-visualization-btn",
                    color="secondary",
                    className="me-2",
                ),
                dbc.Button(
                    [
                        html.I(className="fas fa-save me-2"),
                        "Save Label and Proceed"
                    ],
                    id="save-label-btn",
                    color="success",
                    disabled=True,
                ),
            ], className="d-flex justify-content-between"),
        ]),
    ])

def get_tab4_content():
    return dbc.Container([
        html.H3("Step 4: Review & Submit", className="mb-4"),
        
        # Summary Card
        dbc.Card([
            dbc.CardHeader([
                html.I(className="fas fa-clipboard-list me-2"),
                "Review Summary"
            ], className="fw-bold"),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        html.H5([
                            html.I(className="fas fa-file me-2"),
                            "Selected Files"
                        ], className="mb-3"),
                        html.Div(
                            id="review-selected-files",
                            className="mb-4"
                        ),
                    ], md=6),
                    dbc.Col([
                        html.H5([
                            html.I(className="fas fa-tag me-2"),
                            "Assigned Labels"
                        ], className="mb-3"),
                        html.Div(
                            id="review-assigned-labels",
                            className="mb-4"
                        ),
                    ], md=6),
                ]),
            ]),
        ], className="mb-4"),
        
        # Submit Alert
        dbc.Alert(
            id="submit-alert-message",
            dismissable=True,
            duration=4000,
            is_open=False,
        ),
        
        # Navigation Buttons
        dbc.Row([
            dbc.Col([
                dbc.Button(
                    [
                        html.I(className="fas fa-arrow-left me-2"),
                        "Back to Labeling"
                    ],
                    id="back-to-labeling-btn",
                    color="secondary",
                    className="me-2",
                ),
                dbc.Button(
                    [
                        html.I(className="fas fa-check-circle me-2"),
                        "Submit Labels"
                    ],
                    id="submit-labels-btn",
                    color="success",
                    disabled=True,
                ),
            ], className="d-flex justify-content-between"),
        ]),
    ])