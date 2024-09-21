import dash
from dash import html, dcc, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import os
import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt
from scipy.fft import fft
from flask import send_file, request

import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend suitable for scripts
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# High-pass filter function
def high_pass_filter(data, cutoff=0.1, fs=100.0, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    y = filtfilt(b, a, data)
    return y

# FFT calculation function
def calculate_fft(data, fs=100.0):
    N = len(data)
    T = 1.0 / fs
    yf = fft(data)
    xf = np.linspace(0.0, 1.0 / (2.0 * T), N // 2)
    amplitudes = 2.0 / N * np.abs(yf[:N // 2])
    return xf, amplitudes

# RMS calculation function
def calculate_rms(series):
    return np.sqrt(np.mean(series**2))

# Feature extraction function
def extract_all_features(data, cutoff=0.1, fs=100.0, order=5):
    features = {}
    try:
        # Basic statistical features
        features = {
            'Mean': np.round(data.mean(), 3),
            'Std Dev': np.round(data.std(), 3),
            'RMS': np.round(np.sqrt(np.mean(data ** 2)), 3),
            'Max': np.round(data.max(), 4),
            'Min': np.round(data.min(), 4)
        }

        # High-pass filter
        y = high_pass_filter(data.to_numpy(), cutoff, fs, order)

        # High-pass filtered features
        features.update({
            'HPF Std Dev': np.round(y.std(), 4),
            'HPF Max': np.round(y.max(), 4),
            'HPF Min': np.round(y.min(), 4),
            'HPF RMS': np.round(np.sqrt(np.mean(y ** 2)), 4)
        })

        # FFT features
        N = len(data)
        yf = fft(data.to_numpy())
        abs_yf = np.abs(yf[:N // 2])
        features.update({
            'FFT Mean': np.round(np.mean(abs_yf), 4),
            'FFT Std Dev': np.round(np.std(abs_yf), 4),
            'FFT Max': np.round(np.max(abs_yf), 4)
        })
        return features
    except Exception as e:
        print(f"Error in feature extraction: {e}")
        return {}

# Matplotlib plot functions
def plot_data_matplotlib(data, x_col, y_col, title, ax):
    ax.plot(data[x_col], data[y_col])
    ax.axhline(y=1, color='r', linestyle='-')
    ax.axhline(y=9, color='r', linestyle='-')
    ax.set_ylim(0, 10)
    ax.set_yticks(range(0, 11))
    ax.set_xlabel('X[mm]')
    ax.set_ylabel('N[Ncm]')
    ax.set_title(title)

def plot_fft_matplotlib(data, ax):
    N = len(data)
    fs = 100.0
    T = 1.0 / fs
    yf = fft(data)
    xf = np.linspace(0.0, 1.0 / (2.0 * T), N // 2)
    ax.set_ylim(0, 0.05)
    ax.stem(xf, 2.0 / N * np.abs(yf[:N // 2]), markerfmt=" ", basefmt="-b", linefmt="-")
    ax.set_title('FFT of Data')
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel('Amplitude')

def add_features_text_matplotlib(ax, features, position=[0, 1]):
    feature_text = '\n'.join([f'{key}: {value}' for key, value in features.items()])
    ax.text(position[0], position[1], feature_text, transform=ax.transAxes, fontsize=10, va='top', ha='left')

# PDF generation function
def generate_pdf(csv_file_path):
    try:
        # Read CSV file and clean column names
        data = pd.read_csv(csv_file_path, encoding='shift-jis')
        data.columns = [col.strip() for col in data.columns]

        # Ensure required columns are present
        required_columns = ['X[mm]', 'N[Ncm]']
        if not all(col in data.columns for col in required_columns):
            print(f"Required columns missing in {csv_file_path}. Found columns: {data.columns}")
            return

        # Extract features
        features = extract_all_features(data['N[Ncm]'])

        # Creating PDF file
        output_pdf_path = csv_file_path.replace('.csv', '.pdf')
        with PdfPages(output_pdf_path) as pdf:
            fig, axes = plt.subplots(4, 1, figsize=(8.27, 11.69),
                                     gridspec_kw={'height_ratios': [1, 1, 1, 0.3]})
            plt.subplots_adjust(left=0.1, right=0.9, top=0.95, bottom=0.05, hspace=0.4)

            # Plot 1: Normal data plot
            plot_data_matplotlib(data, 'X[mm]', 'N[Ncm]', csv_file_path.replace('.csv', ''), axes[0])

            # Plot 2: High-pass filtered data plot
            filtered_data = high_pass_filter(data['N[Ncm]'].to_numpy(), cutoff=0.1)
            filtered_series = pd.Series(filtered_data, index=data.index)
            filtered_rms = filtered_series.rolling(window=200).apply(calculate_rms, raw=True)
            axes[1].plot(data['X[mm]'], filtered_data, label='Filtered Data')
            axes[1].plot(data['X[mm]'], filtered_rms, 'r', label='RMS')
            axes[1].set_ylim(-1, 1)
            axes[1].set_xlabel('X[mm]')
            axes[1].set_ylabel('N[Ncm]')
            axes[1].set_title('High-pass Filtered Data (Cutoff = 0.1 Hz)')
            axes[1].legend()

            # Plot 3: FFT plot
            plot_fft_matplotlib(data['N[Ncm]'].to_numpy(), axes[2])

            # Add features as text below the plots
            add_features_text_matplotlib(axes[3], features, position=[0, 1])
            axes[3].axis('off')

            plt.tight_layout()

            # Save the figure to the PDF
            pdf.savefig(fig)
            plt.close(fig)

        print(f"Processed and saved PDF for {csv_file_path}")
    except Exception as e:
        print(f"Error processing CSV file {csv_file_path}: {e}")

# App layout
app.layout = dbc.Container([
    html.H1("Torque Measurement Visualization"),

    # File Selection
    dbc.Row([
        dbc.Col([
            html.Label("Select Measurement File:"),
            dcc.Dropdown(id='file-dropdown', options=[], placeholder="Select a CSV file"),
        ], width=6),
    ]),

    # High-Pass Filter Cutoff Input
    dbc.Row([
        dbc.Col([
            html.Label("High-Pass Filter Cutoff Frequency (Hz):"),
            dcc.Input(id='cutoff-input', type='number', value=0.1, step=0.1),
        ], width=3),
    ]),

    # Axis Scaling Inputs
    dbc.Row([
        dbc.Col([
            html.Label("Y-axis Scale:"),
            dcc.RangeSlider(id='y-axis-slider', min=0, max=10, step=0.5, value=[0, 10]),
        ], width=6),
    ]),

    # Graphs
    dbc.Row([
        dbc.Col([
            dcc.Graph(id='normal-graph'),
            dcc.Graph(id='filtered-graph'),
            dcc.Graph(id='fft-graph'),
        ], width=12),
    ]),

    # Download Links
    dbc.Row([
        dbc.Col([
            html.A("Download CSV", id='download-csv', href="", target="_blank"),
            html.Br(),
            html.A("Download PDF", id='download-pdf', href="", target="_blank"),
        ]),
    ]),
])

# Callback to update file options
@app.callback(
    Output('file-dropdown', 'options'),
    Input('file-dropdown', 'id')  # Dummy input to trigger on page load
)
def update_file_options(_):
    directory = 'W:/'
    files = [f for f in os.listdir(directory) if f.endswith('.csv')]
    options = [{'label': f, 'value': os.path.join(directory, f)} for f in files]
    return options

# Callback to update graphs and download links
@app.callback(
    [Output('normal-graph', 'figure'),
     Output('filtered-graph', 'figure'),
     Output('fft-graph', 'figure'),
     Output('download-csv', 'href'),
     Output('download-pdf', 'href')],
    [Input('file-dropdown', 'value'),
     Input('cutoff-input', 'value'),
     Input('y-axis-slider', 'value')]
)
def update_graphs(file_path, cutoff_freq, y_axis_range):
    if file_path is None:
        return {}, {}, {}, "", ""

    # Read data
    data = pd.read_csv(file_path, encoding='shift-jis')
    data.columns = [col.strip() for col in data.columns]
    x = data['X[mm]']
    y = data['N[Ncm]']

    # Normal Plot
    normal_fig = go.Figure()
    normal_fig.add_trace(go.Scatter(x=x, y=y, mode='lines'))
    normal_fig.update_layout(
        title='Normal Data Plot',
        xaxis_title='X[mm]',
        yaxis_title='N[Ncm]',
        yaxis=dict(range=y_axis_range),
    )

    # High-Pass Filtered Plot
    y_filtered = high_pass_filter(y.to_numpy(), cutoff=cutoff_freq)
    filtered_fig = go.Figure()
    filtered_fig.add_trace(go.Scatter(x=x, y=y_filtered, mode='lines', name='Filtered Data'))
    # Rolling RMS
    filtered_series = pd.Series(y_filtered)
    filtered_rms = filtered_series.rolling(window=200).apply(lambda s: np.sqrt(np.mean(s**2)), raw=True)
    filtered_fig.add_trace(go.Scatter(x=x, y=filtered_rms, mode='lines', name='RMS'))
    filtered_fig.update_layout(
        title=f'High-Pass Filtered Data (Cutoff = {cutoff_freq} Hz)',
        xaxis_title='X[mm]',
        yaxis_title='N[Ncm]',
        yaxis=dict(range=[-1, 1]),
    )

    # FFT Plot
    xf, amplitudes = calculate_fft(y.to_numpy())
    fft_fig = go.Figure()
    fft_fig.add_trace(go.Scatter(x=xf, y=amplitudes, mode='lines'))
    fft_fig.update_layout(
        title='FFT of Data',
        xaxis_title='Frequency [Hz]',
        yaxis_title='Amplitude',
        yaxis=dict(range=[0, 0.05]),
    )

    # Download Links
    csv_href = f'/download/csv?filepath={file_path}'
    pdf_href = f'/download/pdf?filepath={file_path.replace(".csv", ".pdf")}'

    return normal_fig, filtered_fig, fft_fig, csv_href, pdf_href

# Route to download CSV files
@app.server.route('/download/csv')
def download_csv():
    file_path = request.args.get('filepath')
    return send_file(file_path,
                     mimetype='text/csv',
                     attachment_filename=os.path.basename(file_path),
                     as_attachment=True)

# Route to download PDF files
@app.server.route('/download/pdf')
def download_pdf():
    file_path = request.args.get('filepath')
    if not os.path.exists(file_path):
        # Generate PDF if it doesn't exist
        csv_file_path = file_path.replace('.pdf', '.csv')
        generate_pdf(csv_file_path)
    return send_file(file_path,
                     mimetype='application/pdf',
                     attachment_filename=os.path.basename(file_path),
                     as_attachment=True)

if __name__ == '__main__':
    #app.run_server(debug=True, host='0.0.0.0', port=8050)
    app.run_server(debug=True)
