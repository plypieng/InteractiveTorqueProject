# app/processing/feature_extraction.py
import numpy as np
from scipy.fft import fft
from .filters import high_pass_filter
import logging

def calculate_rms(series):
    return np.sqrt(np.mean(series**2))

def calculate_fft(data, fs=100.0):
    try:
        N = len(data)
        T = 1.0 / fs
        yf = fft(data)
        xf = np.linspace(0.0, 1.0 / (2.0 * T), N // 2)
        amplitudes = 2.0 / N * np.abs(yf[:N // 2])
        return xf, amplitudes
    except Exception as e:
        logging.error(f"Error in calculate_fft: {e}")
        raise e

def extract_all_features(data, cutoff=0.1, fs=100.0, order=5):
    features = {}
    try:
        # Statistical Features
        features = {
            'Mean': np.round(data.mean(), 3),
            'Std Dev': np.round(data.std(), 3),
            'RMS': np.round(np.sqrt(np.mean(data ** 2)), 3),
            'Max': np.round(data.max(), 4),
            'Min': np.round(data.min(), 4)
        }

        # High-Pass Filter
        y = high_pass_filter(data.to_numpy(), cutoff, fs, order)

        # HPF Statistical Features
        features.update({
            'HPF Std Dev': np.round(y.std(), 4),
            'HPF Max': np.round(y.max(), 4),
            'HPF Min': np.round(y.min(), 4),
            'HPF RMS': np.round(np.sqrt(np.mean(y ** 2)), 4),
        })

        # FFT Features
        xf, amplitudes = calculate_fft(data.to_numpy(), fs)
        features.update({
            'FFT Mean': np.round(np.mean(amplitudes), 4),
            'FFT Std Dev': np.round(np.std(amplitudes), 4),
            'FFT Max': np.round(np.max(amplitudes), 4)
        })
        return features
    except Exception as e:
        logging.error(f"Error in extract_all_features: {e}")
        return {}
