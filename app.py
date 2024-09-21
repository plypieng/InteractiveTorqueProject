# app.py

import dash
from dash import html, dcc, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import os
import pandas as pd
from flask import send_file, request, abort
from dash.exceptions import PreventUpdate
import logging
from urllib.parse import quote_plus, unquote_plus

from utils import high_pass_filter, calculate_fft, calculate_rms, extract_all_features, load_data, is_safe_path
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
- **Adjust High-Pass Filter Cutoff Frequency:** Enter a value in Hz.
- **Adjust Y-axis Scale:** Use the slider to set the range.
- **View Graphs:** The plots will update based on your selections.
- **View Extracted Features:** Features from the data are displayed below the graphs.
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

    # High-Pass Filter Cutoff Input
    dbc.Row([
        dbc.Col([
            html.Label("High-Pass Filter Cutoff Frequency (Hz):"),
            dcc.Input(id='cutoff-input', type='number', value=0.1, step=0.1),
        ], width=3),
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
    dcc.Loading(
        id="loading",
        type="default",
        children=[
            # Graphs for File 1
            html.H3("File 1 Graphs", className="text-center"),
            dbc.Row([
                dbc.Col([dcc.Graph(id='normal-graph-1')], width=6),
                dbc.Col([dcc.Graph(id='filtered-graph-1')], width=6),
            ]),
            dbc.Row([
                dbc.Col([dcc.Graph(id='fft-graph-1')], width=6),
                dbc.Col([
                    html.H5("Extracted Features for File 1:"),
                    html.Div(id='features-1', style={'whiteSpace': 'pre-wrap'}),
                ], width=6),
            ]),

            html.Hr(),

            # Graphs for File 2 (if selected)
            html.H3("File 2 Graphs", className="text-center"),
            dbc.Row([
                dbc.Col([dcc.Graph(id='normal-graph-2')], width=6),
                dbc.Col([dcc.Graph(id='filtered-graph-2')], width=6),
            ]),
            dbc.Row([
                dbc.Col([dcc.Graph(id='fft-graph-2')], width=6),
                dbc.Col([
                    html.H5("Extracted Features for File 2:"),
                    html.Div(id='features-2', style={'whiteSpace': 'pre-wrap'}),
                ], width=6),
            ]),
        ]
    ),

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
    Output('file-dropdown-2', 'options'),
    Input('file-dropdown-1', 'id')  # Dummy input to trigger on page load
)
def update_file_options(_):
    directory = ALLOWED_DIRECTORY
    try:
        files = [f for f in os.listdir(directory) if f.endswith('.csv')]
        options = [{'label': f, 'value': os.path.join(directory, f)} for f in files]
        return options, options
    except Exception as e:
        logging.error(f"Error accessing directory {directory}: {e}")
        return [], []

# Callback to update graphs and download links
@app.callback(
    [Output('normal-graph-1', 'figure'),
     Output('filtered-graph-1', 'figure'),
     Output('fft-graph-1', 'figure'),
     Output('features-1', 'children'),
     Output('download-csv-1', 'href'),
     Output('download-pdf-1', 'href'),
     Output('normal-graph-2', 'figure'),
     Output('filtered-graph-2', 'figure'),
     Output('fft-graph-2', 'figure'),
     Output('features-2', 'children'),
     Output('download-csv-2', 'href'),
     Output('download-pdf-2', 'href')],
    [Input('file-dropdown-1', 'value'),
     Input('file-dropdown-2', 'value'),
     Input('cutoff-input', 'value'),
     Input('y-axis-slider', 'value')]
)
def update_graphs(file_path_1, file_path_2, cutoff_freq, y_axis_range):
    outputs = []

    # Process File 1
    if file_path_1:
        figs_1, features_text_1, csv_href_1, pdf_href_1 = process_file(file_path_1, cutoff_freq, y_axis_range, '1')
    else:
        figs_1 = [{}, {}, {}]
        features_text_1 = ""
        csv_href_1 = ""
        pdf_href_1 = ""

    # Process File 2
    if file_path_2:
        figs_2, features_text_2, csv_href_2, pdf_href_2 = process_file(file_path_2, cutoff_freq, y_axis_range, '2')
    else:
        figs_2 = [{}, {}, {}]
        features_text_2 = ""
        csv_href_2 = ""
        pdf_href_2 = ""

    # Combine outputs
    outputs.extend(figs_1)
    outputs.append(features_text_1)
    outputs.append(csv_href_1)
    outputs.append(pdf_href_1)
    outputs.extend(figs_2)
    outputs.append(features_text_2)
    outputs.append(csv_href_2)
    outputs.append(pdf_href_2)

    return outputs

def process_file(file_path, cutoff_freq, y_axis_range, file_label):
    # Security check for file path
    if not is_safe_path(ALLOWED_DIRECTORY, file_path):
        logging.warning(f"Attempt to access invalid file path: {file_path}")
        error_fig = go.Figure()
        error_fig.add_annotation(text="Invalid file path selected.",
                                 xref="paper", yref="paper",
                                 showarrow=False, font=dict(color="red", size=16))
        return [error_fig, {}, {}], "", "", ""

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
        return [error_fig, {}, {}], "", "", ""

    # Generate plots
    normal_fig = create_normal_plot(x, y, y_axis_range)
    y_filtered = high_pass_filter(y.to_numpy(), cutoff=cutoff_freq)
    filtered_series = pd.Series(y_filtered)
    filtered_rms = filtered_series.rolling(window=500).apply(calculate_rms, raw=True)
    filtered_fig = create_filtered_plot(x, y_filtered, filtered_rms, cutoff_freq)
    xf, amplitudes = calculate_fft(y.to_numpy())
    fft_fig = create_fft_plot(xf, amplitudes)

    # Extract features
    features = extract_all_features(data['N[Ncm]'], cutoff=cutoff_freq)
    features_text = '\n'.join([f'{key}: {value}' for key, value in features.items()])

    # URL-encode file paths
    encoded_file_path = quote_plus(file_path)
    encoded_pdf_path = quote_plus(file_path.replace('.csv', '.pdf'))

    # Download Links with cutoff frequency parameter
    csv_href = f'/download/csv?filepath={encoded_file_path}'
    pdf_href = f'/download/pdf?filepath={encoded_pdf_path}&cutoff={cutoff_freq}'

    return [normal_fig, filtered_fig, fft_fig], features_text, csv_href, pdf_href

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
