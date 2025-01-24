# app/processing/feature_extraction.py
import numpy as np
import pandas as pd
from scipy.fft import fft, fftfreq
from scipy.signal import butter, filtfilt, medfilt
from scipy.stats import median_abs_deviation
from scipy import stats
import logging

from .filters import high_pass_filter, apply_filter

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
        logging.error(f"Error in calculate_fft: {e}", exc_info=True)
        raise e

# Deprecated
def extract_all_features(data, cutoff=0.1, fs=100.0, order=5):
    """
    Old method used by measurement_tasks. 
    """
    features = {}
    try:
        features = {
            'Mean': np.round(data.mean(), 3),
            'Std Dev': np.round(data.std(), 3),
            'RMS': np.round(np.sqrt(np.mean(data ** 2)), 3),
            'Max': np.round(data.max(), 4),
            'Min': np.round(data.min(), 4)
        }
        y = high_pass_filter(data.to_numpy(), cutoff, fs, order)
        features.update({
            'HPF Std Dev': np.round(y.std(), 4),
            'HPF Max': np.round(y.max(), 4),
            'HPF Min': np.round(y.min(), 4),
            'HPF RMS': np.round(np.sqrt(np.mean(y ** 2)), 4),
        })

        xf, amplitudes = calculate_fft(data.to_numpy(), fs)
        features.update({
            'FFT Mean': np.round(np.mean(amplitudes), 4),
            'FFT Std Dev': np.round(np.std(amplitudes), 4),
            'FFT Max': np.round(np.max(amplitudes), 4)
        })
        return features
    except Exception as e:
        logging.error(f"Error in extract_all_features: {e}", exc_info=True)
        return {}

def remove_spikes(data, threshold=7.5):
    return np.clip(data, -threshold, threshold)

def calculate_entropy(signal_data):
    histogram, _ = np.histogram(signal_data, bins=32, density=True)
    histogram += 1e-12
    return -np.sum(histogram * np.log2(histogram))

def calculate_spectral_entropy(signal_data, fs=100.0):
    fft_vals = np.abs(fft(signal_data))
    fft_vals = fft_vals[:len(fft_vals)//2]
    fft_vals = fft_vals / np.sum(fft_vals)
    fft_vals += 1e-12
    return -np.sum(fft_vals * np.log2(fft_vals))

def extract_features(torque, fs=100.0, window_size=300, threshold=7.5, kernel_size=1):
    """
    New function to extract rich time-domain & freq-domain features.
    """
    filtered_torque = apply_filter(torque, type='high', cutoff=10, fs=fs, order=5)
    torque_cleaned = remove_spikes(filtered_torque, threshold=threshold)
    torque_filtered = medfilt(torque_cleaned, kernel_size=kernel_size)
    filtered_series = pd.Series(torque_filtered)

    mean_val = np.mean(torque_filtered)
    median_value = np.median(torque_filtered)
    mad_val = median_abs_deviation(torque_filtered)
    std_val = np.std(torque_filtered)
    rms_val = np.sqrt(np.mean(torque_filtered**2))
    max_value = np.max(torque_filtered)
    min_value = np.min(torque_filtered)
    peak = max(abs(max_value), abs(min_value))
    mean_abs = np.mean(np.abs(torque_filtered))
    skewness = stats.skew(torque_filtered)
    kurtosis_val = stats.kurtosis(torque_filtered, fisher=False)

    shape_factor = rms_val / mean_abs if mean_abs != 0 else 0
    crest_factor = peak / rms_val if rms_val != 0 else 0
    signal_entropy = calculate_entropy(torque_filtered)

    gradient = np.gradient(torque_filtered)
    gradient_mean = np.mean(gradient)
    gradient_std = np.std(gradient)

    rolling_median = filtered_series.rolling(window_size, min_periods=1).median()
    rolling_mad = filtered_series.rolling(window_size, min_periods=1).apply(median_abs_deviation)

    N = len(torque_filtered)
    freqs = fftfreq(N, 1/fs)
    fft_vals = fft(torque_filtered)
    fft_magnitude = np.abs(fft_vals)

    pos_mask = freqs >= 0
    freqs = freqs[pos_mask]
    fft_magnitude = fft_magnitude[pos_mask]

    spectral_centroid = np.sum(freqs * fft_magnitude) / np.sum(fft_magnitude)
    spectral_ent = calculate_spectral_entropy(torque_filtered, fs)
    peak_frequency = freqs[np.argmax(fft_magnitude)]
    spectral_flatness = (np.exp(np.mean(np.log(fft_magnitude))) / np.mean(fft_magnitude)
                         if np.mean(fft_magnitude) != 0 else 0)
    spectral_spread = np.sqrt(
        np.sum(((freqs - spectral_centroid) ** 2) * fft_magnitude) / np.sum(fft_magnitude)
    )
    cumulative_spectrum = np.cumsum(fft_magnitude)
    roll_off_level = 0.85 * np.sum(fft_magnitude)
    spectral_roll_off = freqs[np.where(cumulative_spectrum >= roll_off_level)][0]
    low_band_energy = np.sum(fft_magnitude[(freqs >= 0) & (freqs < 10)])
    mid_band_energy = np.sum(fft_magnitude[(freqs >= 10) & (freqs < 30)])
    high_band_energy = np.sum(fft_magnitude[(freqs >= 30)])
    spectral_crest = (np.max(fft_magnitude) / np.mean(fft_magnitude)
                      if np.mean(fft_magnitude) != 0 else 0)
    flux = np.diff(fft_magnitude)
    spectral_flux = np.sum(flux**2) if len(flux) > 0 else 0
    spectral_kurtosis = stats.kurtosis(fft_magnitude)
    spectral_skewness = stats.skew(fft_magnitude)
    spectral_slope = np.polyfit(freqs, fft_magnitude, 1)[0] if len(freqs) > 1 else 0

    features = {
        'Mean': mean_val,
        'Median': median_value,
        'MAD': mad_val,
        'Standard Deviation': std_val,
        'RMS': rms_val,
        'Shape Factor': shape_factor,
        'Crest Factor': crest_factor,
        'Entropy': signal_entropy,
        'Skewness': skewness,
        'Kurtosis': kurtosis_val,
        'Gradient Mean': gradient_mean,
        'Gradient Std Dev': gradient_std,
        'Rolling Median Mean': rolling_median.mean(),
        'Rolling MAD Mean': rolling_mad.mean(),
        'Spectral Centroid': spectral_centroid,
        'Spectral Entropy': spectral_ent,
        'Peak Frequency': peak_frequency,
        'Spectral Flatness': spectral_flatness,
        'Spectral Spread': spectral_spread,
        'Spectral Roll-off': spectral_roll_off,
        'Low Band Energy': low_band_energy,
        'Mid Band Energy': mid_band_energy,
        'High Band Energy': high_band_energy,
        'Spectral Crest': spectral_crest,
        'Spectral Flux': spectral_flux,
        'Spectral Kurtosis': spectral_kurtosis,
        'Spectral Skewness': spectral_skewness,
        'Spectral Slope': spectral_slope
    }
    return features
