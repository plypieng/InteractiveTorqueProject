# pdf_generator.py

import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend suitable for scripts
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from utils import high_pass_filter, calculate_rms, extract_all_features
import logging
import numpy as np
from scipy.fft import fft

def plot_data_matplotlib(data, x_col, y_col, title, ax):
    """
    Plot the normal data using Matplotlib.

    Parameters:
    - data (pd.DataFrame): DataFrame containing the data.
    - x_col (str): Column name for X-axis.
    - y_col (str): Column name for Y-axis.
    - title (str): Plot title.
    - ax (matplotlib.axes.Axes): Matplotlib Axes object.
    """
    ax.plot(data[x_col], data[y_col])
    ax.axhline(y=1, color='r', linestyle='-')
    ax.axhline(y=9, color='r', linestyle='-')
    ax.set_ylim(0, 10)
    ax.set_yticks(range(0, 11))
    ax.set_xlabel('X[mm]')
    ax.set_ylabel('N[Ncm]')
    ax.set_title(title)

def plot_fft_matplotlib(data, ax):
    """
    Plot the FFT of the data using Matplotlib.

    Parameters:
    - data (array-like): Input data.
    - ax (matplotlib.axes.Axes): Matplotlib Axes object.
    """
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
    """
    Add text annotations of features to the plot.

    Parameters:
    - ax (matplotlib.axes.Axes): Matplotlib Axes object.
    - features (dict): Dictionary of features.
    - position (list): Position of the text [x, y].
    """
    feature_text = '\n'.join([f'{key}: {value}' for key, value in features.items()])
    ax.text(position[0], position[1], feature_text, transform=ax.transAxes, fontsize=10, va='top', ha='left')

def generate_pdf(csv_file_path, cutoff=0.1):
    """
    Generate a PDF report from the CSV data.

    Parameters:
    - csv_file_path (str): Path to the CSV file.
    - cutoff (float): Cutoff frequency for high-pass filter.
    """
    try:
        # Read CSV file and clean column names
        data = pd.read_csv(csv_file_path, encoding='shift-jis')
        data.columns = [col.strip() for col in data.columns]

        # Ensure required columns are present
        required_columns = ['X[mm]', 'N[Ncm]']
        if not all(col in data.columns for col in required_columns):
            logging.error(f"Required columns missing in {csv_file_path}. Found columns: {data.columns}")
            return

        # Extract features
        features = extract_all_features(data['N[Ncm]'], cutoff=cutoff)

        # Creating PDF file
        output_pdf_path = csv_file_path.replace('.csv', '.pdf')
        with PdfPages(output_pdf_path) as pdf:
            fig, axes = plt.subplots(4, 1, figsize=(8.27, 11.69),
                                     gridspec_kw={'height_ratios': [1, 1, 1, 0.3]})
            plt.subplots_adjust(left=0.1, right=0.9, top=0.95, bottom=0.05, hspace=0.4)

            # Plot 1: Normal data plot
            plot_data_matplotlib(data, 'X[mm]', 'N[Ncm]', csv_file_path.replace('.csv', ''), axes[0])

            # Plot 2: High-pass filtered data plot
            filtered_data = high_pass_filter(data['N[Ncm]'].to_numpy(), cutoff=cutoff)
            filtered_series = pd.Series(filtered_data, index=data.index)
            filtered_rms = filtered_series.rolling(window=200).apply(calculate_rms, raw=True)
            axes[1].plot(data['X[mm]'], filtered_data, label='Filtered Data')
            axes[1].plot(data['X[mm]'], filtered_rms, 'r', label='RMS')
            axes[1].set_ylim(-1, 1)
            axes[1].set_xlabel('X[mm]')
            axes[1].set_ylabel('N[Ncm]')
            axes[1].set_title(f'High-pass Filtered Data (Cutoff = {cutoff} Hz)')
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

        logging.info(f"Processed and saved PDF for {csv_file_path}")
    except Exception as e:
        logging.error(f"Error processing CSV file {csv_file_path}: {e}")
