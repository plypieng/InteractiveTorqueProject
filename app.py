# app.py

import dash
from dash import html, dcc, Input, Output, State, no_update
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import os
import pandas as pd
from flask import send_file, request, abort
from dash.exceptions import PreventUpdate
import logging
from urllib.parse import quote_plus, unquote_plus

from utils import (high_pass_filter, calculate_fft, calculate_rms, extract_all_features, 
                   load_data, is_safe_path, detected_sudden_spike, analyse_hpf_rms)
from plots import create_normal_plot, create_filtered_plot, create_fft_plot
from pdf_generator import generate_pdf

# Configure logging
logging.basicConfig(level=logging.INFO, filename='app.log',
                    format='%(asctime)s %(levelname)s:%(message)s')

# Allowed directory for file access
ALLOWED_DIRECTORY = 'W:/'

# Initialize Dash app with a Bootstrap theme
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.SOLAR])

# Help text for the user guide modal
help_text = '''
### How to Use This Application
- **Select Measurement Files:** Choose up to two CSV files from the dropdowns to compare.
- **Adjust Parameters:**
  - **High-Pass Filter Cutoff Frequency:** Enter a value in Hz.
  - **RMS Window Size:** Enter the window size for calculating moving RMS.
  - **HPF_RMS Threshold:** Set the threshold for HPF_RMS.
  - **Spike Threshold:** Set the threshold for spike detection in RMS data.
- **Adjust Y-axis Scale:** Use the slider to set the range.
- **View Graphs:** The plots will update based on your selections.
- **View Analysis Results:** PASS or FAILED results are displayed based on the analysis.
- **Download Data:** Use the links to download CSV or PDF files.
'''

# App layout
app.layout = dbc.Container([
    html.H1("Torque Measurement Visualization", className="text-center text-primary mb-4"),

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

    # Interval component for updating file options
    dcc.Interval(
        id='interval-component',
        interval=2*1000,  # Update every 10 seconds
        n_intervals=0
    ),

    # File Selection for File 1
    dbc.Row([
        dbc.Col([
            html.Label("Select Measurement File 1:"),
            dcc.Dropdown(id='file-dropdown-1', options=[], placeholder="Select a CSV file"),
        ], width=6),
    ], className="mb-3"),

    # File Selection for File 2
    dbc.Row([
        dbc.Col([
            html.Label("Select Measurement File 2 (optional):"),
            dcc.Dropdown(id='file-dropdown-2', options=[], placeholder="Select a CSV file"),
        ], width=6),
    ], className="mb-3"),

    # parameter Input
    dbc.Row([
        dbc.Col([
            html.Label("High-Pass Filter Cutoff Frequency (Hz):"),
            dcc.Input(id='cutoff-input', type='number', value=10, step=0.1),
        ], width=3),
        dbc.Col([
            html.Label("RMS Window Size (samples):"),
            dcc.Input(id='rms-window-size', type='number', value=300, step=100),
        ], width=3),
        dbc.Col([
            html.Label("hpf-rms-Threshold (Ncm):"),
            dcc.Input(id='hpf-rms-threshold', type='number', value=0.05, step=0.005),
        ], width=3),
        dbc.Col([
            html.Label("Spike Threshold (Ncm):"),
            dcc.Input(id='spike-threshold', type='number', value=0.01, step=0.001),
        ])
    ], className="mb-3"),

    # Axis Scaling Inputs
    dbc.Row([
        dbc.Col([
            html.Label("Y-axis Scale:"),
            dcc.RangeSlider(
                id='y-axis-slider',
                min=0, max=10, step=0.5, value=[0, 10],
                marks={i: str(i) for i in range(0, 11)}
            ),
        ], width=6),
    ], className="mb-4"),

    # Loading indicator and Graphs
    # Graphs for File 1
    html.H3("File 1 Analysis", className="text-center"),
    dbc.Row([
        dbc.Col([
            dcc.Loading(
                id='loading-normal-graph-1',
                type='default',
                children=dcc.Graph(id='normal-graph-1'),
                style={'display': 'inline-block'},
                overlay_style={"visibility": "visible"}
            )
        ], width=6),
        dbc.Col([
            dcc.Loading(
                id='loading-filtered-graph-1',
                type='default',
                children=dcc.Graph(id='filtered-graph-1'),
                style={'display': 'inline-block'},
            )
        ], width=6),
    ]),
    dbc.Row([
        dbc.Col([
            dcc.Loading(
                id='loading-fft-graph-1',
                type='default',
                children=dcc.Graph(id='fft-graph-1'),
                style={'display': 'inline-block'},
                overlay_style={"visibility": "visible"}
            )
        ], width=6),
        dbc.Col([
            html.H5("Extracted Features for File 1:"),
            html.Div(id='features-1', style={'whiteSpace': 'pre-wrap'}),
            html.H5("Analysis Result for File 1:"),
            html.Div(id='analysis-result-1', style={'whiteSpace': 'pre-wrap', 'fontWeight': 'bold'}),
        ], width=6),
    ]),

    html.Hr(),

    # Graphs for File 2 (if selected)
    html.H3("File 2 Analysis", className="text-center"),
    dbc.Row([
        dbc.Col([
            dcc.Loading(
                id='loading-normal-graph-2',
                type='default',
                children=dcc.Graph(id='normal-graph-2'),
                style={'display': 'inline-block'},
                overlay_style={"visibility": "visible"}
            )
        ], width=6),
        dbc.Col([
            dcc.Loading(
                id='loading-filtered-graph-2',
                type='default',
                children=dcc.Graph(id='filtered-graph-2'),
                style={'display': 'inline-block'},
            )
        ], width=6),
    ]),
    dbc.Row([
        dbc.Col([
            dcc.Loading(
                id='loading-fft-graph-2',
                type='default',
                children=dcc.Graph(id='fft-graph-2'),
                style={'display': 'inline-block'},
                overlay_style={"visibility": "visible"}
            )
        ], width=6),
        dbc.Col([
            html.H5("Extracted Features for File 2:"),
            html.Div(id='features-2', style={'whiteSpace': 'pre-wrap'}),
            html.H5("Analysis Result for File 2:"),
            html.Div(id='analysis-result-2', style={'whiteSpace': 'pre-wrap', 'fontWeight': 'bold'}),
        ], width=6),
    ]),

    # Download Links
    dbc.Row([
        dbc.Col([
            html.H5("Download Files for File 1:"),
            html.A("Download CSV", id='download-csv-1', href="", target="_blank", className="btn btn-primary mr-2"),
            html.A("Download PDF", id='download-pdf-1', href="", target="_blank", className="btn btn-secondary"),
        ], width=6),
        dbc.Col([
            html.H5("Download Files for File 2:"),
            html.A("Download CSV", id='download-csv-2', href="", target="_blank", className="btn btn-primary mr-2"),
            html.A("Download PDF", id='download-pdf-2', href="", target="_blank", className="btn btn-secondary"),
        ], width=6),
    ], className="mt-4"),
])

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

# Callback to update file options
@app.callback(
    Output('file-dropdown-1', 'options'),
    Output('file-dropdown-1', 'value'),
    Output('file-dropdown-2', 'options'),
    Output('file-dropdown-2', 'value'),
    Input('interval-component', 'n_intervals'),
    State('file-dropdown-1', 'options'),
    State('file-dropdown-1', 'value'),
    State('file-dropdown-2', 'options'),
    State('file-dropdown-2', 'value')
)
def update_file_options(n_intervals, prev_options1, selected_file1, prev_options2, selected_file2):
    directory = ALLOWED_DIRECTORY
    try:
        files = [f for f in os.listdir(directory) if f.endswith('.csv')]
        options = [{'label': f, 'value': os.path.join(directory, f)} for f in files]

        # Compare new options with previous options
        if options == prev_options1:
            # No change in options
            options_update_1 = no_update
            options_update_2 = no_update
        else:
            # Update options
            options_update_1 = options
            options_update_2 = options

        # Check if selected files are still valid
        valid_values = [option['value'] for option in options]
        value_update_1 = selected_file1 if selected_file1 in valid_values else None
        value_update_2 = selected_file2 if selected_file2 in valid_values else None

        # Only update the values if they have changed
        if value_update_1 == selected_file1:
            value_update_1 = no_update
        if value_update_2 == selected_file2:
            value_update_2 = no_update

        return options_update_1, value_update_1, options_update_2, value_update_2
    except Exception as e:
        logging.error(f"Error accessing directory {directory}: {e}")
        return no_update, no_update, no_update, no_update

# Callback to update graphs and download links
@app.callback(
    [Output('normal-graph-1', 'figure'),
     Output('filtered-graph-1', 'figure'),
     Output('fft-graph-1', 'figure'),
     Output('features-1', 'children'),
     Output('analysis-result-1', 'children'),
     Output('download-csv-1', 'href'),
     Output('download-pdf-1', 'href'),
     Output('normal-graph-2', 'figure'),
     Output('filtered-graph-2', 'figure'),
     Output('fft-graph-2', 'figure'),
     Output('features-2', 'children'),
     Output('analysis-result-2', 'children'),
     Output('download-csv-2', 'href'),
     Output('download-pdf-2', 'href')],
    [Input('file-dropdown-1', 'value'),
     Input('file-dropdown-2', 'value'),
     Input('cutoff-input', 'value'),
     Input('rms-window-size', 'value'),
     Input('hpf-rms-threshold', 'value'),
     Input('spike-threshold', 'value'),
     Input('y-axis-slider', 'value')]
)
def update_graphs(file_path_1, file_path_2, cutoff_freq, rms_window_size, hpf_rms_threshold, spike_threshold, y_axis_range):
    outputs = []

    # Process File 1
    if file_path_1:
        figs_1, features_text_1, analysis_result_1, csv_href_1, pdf_href_1 = process_file(
            file_path_1, cutoff_freq, rms_window_size, hpf_rms_threshold, spike_threshold,  y_axis_range, '1')
    else:
        figs_1 = [{}, {}, {}]
        features_text_1 = ""
        analysis_result_1 = ""
        csv_href_1 = ""
        pdf_href_1 = ""

    # Process File 2
    if file_path_2:
        figs_2, features_text_2, analysis_result_2, csv_href_2, pdf_href_2 = process_file(
            file_path_2, cutoff_freq, rms_window_size, hpf_rms_threshold, spike_threshold, y_axis_range, '2')
    else:
        figs_2 = [{}, {}, {}]
        features_text_2 = ""
        analysis_result_2 = ""
        csv_href_2 = ""
        pdf_href_2 = ""

    # Combine outputs
    outputs.extend(figs_1)
    outputs.append(features_text_1)
    outputs.append(analysis_result_1)
    outputs.append(csv_href_1)
    outputs.append(pdf_href_1)
    outputs.extend(figs_2)
    outputs.append(features_text_2)
    outputs.append(analysis_result_2)
    outputs.append(csv_href_2)
    outputs.append(pdf_href_2)

    return outputs

def process_file(file_path, cutoff_freq, rms_window_size, hpf_rms_threshold, spike_threshold, y_axis_range, file_label):
    # Security check for file path
    if not is_safe_path(ALLOWED_DIRECTORY, file_path):
        logging.warning(f"Attempt to access invalid file path: {file_path}")
        error_fig = go.Figure()
        error_fig.add_annotation(text="Invalid file path selected.",
                                 xref="paper", yref="paper",
                                 showarrow=False, font=dict(color="red", size=16))
        return [error_fig, {}, {}], "", "", "", ""

    try:
        # Read data
        data = load_data(file_path)
        x = data['X[mm]']
        y = data['N[Ncm]']
    except Exception as e:
        logging.error(f"Error loading data from {file_path}: {e}")
        error_fig = go.Figure()
        error_fig.add_annotation(text=f"Error loading data: {e}",
                                 xref="paper", yref="paper",
                                 showarrow=False, font=dict(color="red", size=16))
        return [error_fig, {}, {}], "", "", "", ""

    # Generate plots
    normal_fig = create_normal_plot(x, y, y_axis_range)
    y_filtered = high_pass_filter(y.to_numpy(), cutoff=cutoff_freq)
    filtered_series = pd.Series(y_filtered)
    filtered_rms = filtered_series.rolling(window=rms_window_size).apply(calculate_rms, raw=True)
    
    # Compute moving max and min of the filtered data
    moving_max = filtered_series.rolling(window=100, min_periods=1).max()
    moving_min = filtered_series.rolling(window=100, min_periods=1).min()
    
    # Compute moving average of the moving max and min
    moving_max_avg = moving_max.rolling(window=int(rms_window_size), min_periods=1).mean()
    moving_min_avg = moving_min.rolling(window=int(rms_window_size), min_periods=1).mean()

    # Update filtered_fig to include moving max average and moving min average
    filtered_fig = create_filtered_plot(
        x, y_filtered, filtered_rms, moving_max_avg, moving_min_avg, cutoff_freq)
    xf, amplitudes = calculate_fft(y.to_numpy())
    fft_fig = create_fft_plot(xf, amplitudes)

    # Extract features
    features = extract_all_features(data['N[Ncm]'], cutoff=cutoff_freq)
    features_text = '\n'.join([f'{key}: {value}' for key, value in features.items()])

    # Generate analysis result
    analysis_result = []

    # 1. Sudden Spike Detection
    spike_detected = detected_sudden_spike(filtered_rms, spike_threshold)
    if spike_detected:
        analysis_result.append("Sudden spike detected in filtered RMS data. Re-measurement recommended.")
    else:
        analysis_result.append("No sudden spike detected in filtered RMS data.")

    # 2. HPF-RMS Threshold Comparison
    hpf_rms_result = analyse_hpf_rms(filtered_rms, hpf_rms_threshold)
    analysis_result.append(hpf_rms_result)

    # 3. Machine Learning Prediction (Placeholder) need to train some first
    #ml_prediction = machine_learning_prediction(y)
    #analysis_result.append(f"Machine Learning Prediction: {ml_prediction}")

    # Overall Analysis Result (Pass or Failed)
    if spike_detected or "over the threshold" in hpf_rms_result.lower():
        overall_result = "FAILED"
    else:
        overall_result = "PASSED"

    analysis_result_text = '\n'.join(analysis_result)
    analysis_result_text = f"Overall analysis result: {overall_result}\n{analysis_result_text}"

    # URL-encode file paths
    encoded_file_path = quote_plus(file_path)
    encoded_pdf_path = quote_plus(file_path.replace('.csv', '.pdf'))

    # Download Links with cutoff frequency parameter
    csv_href = f'/download/csv?filepath={encoded_file_path}'
    pdf_href = f'/download/pdf?filepath={encoded_pdf_path}&cutoff={cutoff_freq}'

    return [normal_fig, filtered_fig, fft_fig], features_text, analysis_result_text, csv_href, pdf_href

# Route to download CSV files
@app.server.route('/download/csv')
def download_csv():
    try:
        file_path = request.args.get('filepath')
        if not file_path:
            logging.error("File path not provided for CSV download.")
            abort(400, description="File path not provided.")
        # Decode URL-encoded file path
        file_path = unquote_plus(file_path)
        file_path = os.path.normpath(file_path)
        # Security check
        if not is_safe_path(ALLOWED_DIRECTORY, file_path):
            logging.warning(f"Attempt to download invalid file path: {file_path}")
            abort(403, description="Forbidden: Invalid file path.")
        if not os.path.exists(file_path):
            logging.error(f"CSV file not found: {file_path}")
            abort(404, description="CSV file not found.")
        return send_file(file_path,
                         mimetype='text/csv',
                         download_name=os.path.basename(file_path),
                         as_attachment=True)
    except Exception as e:
        logging.error(f"Error in download_csv: {e}")
        abort(500, description="Internal Server Error.")

# Route to download PDF files
@app.server.route('/download/pdf')
def download_pdf():
    try:
        file_path = request.args.get('filepath')
        cutoff = float(request.args.get('cutoff', 0.1))
        if not file_path:
            logging.error("File path not provided for PDF download.")
            abort(400, description="File path not provided.")
        # Decode URL-encoded file path
        file_path = unquote_plus(file_path)
        file_path = os.path.normpath(file_path)
        # Security check
        if not is_safe_path(ALLOWED_DIRECTORY, file_path):
            logging.warning(f"Attempt to download invalid file path: {file_path}")
            abort(403, description="Forbidden: Invalid file path.")
        if not os.path.exists(file_path):
            # Generate PDF if it doesn't exist
            csv_file_path = file_path.replace('.pdf', '.csv')
            if not os.path.exists(csv_file_path):
                logging.error(f"CSV file for PDF generation not found: {csv_file_path}")
                abort(404, description="CSV file for PDF generation not found.")
            generate_pdf(csv_file_path, cutoff)
            if not os.path.exists(file_path):
                logging.error(f"Failed to generate PDF file: {file_path}")
                abort(500, description="Failed to generate PDF file.")
        return send_file(file_path,
                         mimetype='application/pdf',
                         download_name=os.path.basename(file_path),
                         as_attachment=True)
    except Exception as e:
        logging.error(f"Error in download_pdf: {e}")
        abort(500, description="Internal Server Error.")

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True, host='127.0.0.1', port=8050)
