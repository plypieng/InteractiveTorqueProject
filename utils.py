# utils.py

import numpy as np
from scipy.signal import butter, filtfilt
from scipy.fft import fft
import pandas as pd
from functools import lru_cache
import os

def high_pass_filter(data, cutoff=0.1, fs=100.0, order=5):
    """
    Apply a high-pass Butterworth filter to the data.

    Parameters:
    - data (array-like): The input signal data.
    - cutoff (float): The cutoff frequency of the filter in Hz.
    - fs (float): The sampling frequency of the data in Hz.
    - order (int): The order of the filter.

    Returns:
    - y (ndarray): The filtered signal.
    """
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    if normal_cutoff >= 1.0:
        return data  # No filtering if cutoff is too high
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    y = filtfilt(b, a, data)
    return y

def calculate_fft(data, fs=100.0):
    """
    Calculate the Fast Fourier Transform (FFT) of the data.

    Parameters:
    - data (array-like): The input signal data.
    - fs (float): The sampling frequency of the data in Hz.

    Returns:
    - xf (ndarray): Frequencies.
    - amplitudes (ndarray): Corresponding amplitudes.
    """
    N = len(data)
    T = 1.0 / fs
    yf = fft(data)
    xf = np.linspace(0.0, 1.0 / (2.0 * T), N // 2)
    amplitudes = 2.0 / N * np.abs(yf[:N // 2])
    return xf, amplitudes

def calculate_rms(series):
    """
    Calculate the Root Mean Square (RMS) of a series.

    Parameters:
    - series (array-like): Input data series.

    Returns:
    - rms (float): RMS value.
    """
    return np.sqrt(np.mean(series**2))

def extract_all_features(data, cutoff=0.1, fs=100.0, order=5):
    """
    Extract statistical and frequency domain features from the data.

    Parameters:
    - data (pd.Series): Input data series.
    - cutoff (float): Cutoff frequency for high-pass filter.
    - fs (float): Sampling frequency.
    - order (int): Filter order.

    Returns:
    - features (dict): Extracted features.
    """
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
            'HPF RMS': np.round(np.sqrt(np.mean(y ** 2)), 4),
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

@lru_cache(maxsize=32)
def load_data(file_path):
    """
    Load data from a CSV file and cache the result.

    Parameters:
    - file_path (str): Path to the CSV file.

    Returns:
    - data (pd.DataFrame): Loaded data.
    """
    data = pd.read_csv(file_path, encoding='shift-jis')
    data.columns = [col.strip() for col in data.columns]
    return data

def is_safe_path(basedir, path):
    """
    Check if a file path is within a specified base directory.

    Parameters:
    - basedir (str): The base directory.
    - path (str): The file path to check.

    Returns:
    - is_safe (bool): True if the path is safe, False otherwise.
    """
    basedir = os.path.abspath(basedir)
    path = os.path.abspath(path)
    return os.path.commonpath([basedir, path]) == basedir

def detected_sudden_spike(filtered_rms, spike_threshold=0.1):
    """
    Detect sudden spikes in the filtered data.
    
    Parameters:
    - filtered_data (ndarray): The High-pass filtered data.
    - spike_threshold (float): The threshold for detecting spikes.
    
    Returns:
    - spike_detected (bool): True if a sudden spike was detected, False otherwise.
    """

    #filtered_rms = filtered_rms.dropna()
    diff = np.abs(np.diff(filtered_rms))
    spike_detected = np.any(diff > spike_threshold)
    return spike_detected

def analyse_hpf_rms(filtered_rms, threshold):
    """
    Analyse the RMS of the High-pass filtered data.
    
    Parameters:
    - filtered_data (pd.Series): Moving average of the High-pass filtered data.
    - threshold (float): The threshold for HPF_RMS.
    
    Returns:
    - result (str): The result of the analysis.
    """

    average_rms = filtered_rms.mean()
    if average_rms > threshold:
        result = f"HPF_RMS is over the threshold ({threshold}): {average_rms:.4f}"
    else:
        result = f"HPF_RMS is within the threshold ({threshold})."
    return result

#def machine_learning_prediction(data):

#    return "PASS"