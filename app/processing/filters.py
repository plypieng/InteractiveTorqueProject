# app/processing/filters.py
import numpy as np
from scipy.signal import butter, filtfilt
import logging

def high_pass_filter(data, cutoff=0.1, fs=100.0, order=5):
    try:
        nyquist = 0.5 * fs
        normal_cutoff = cutoff / nyquist
        if normal_cutoff >= 1.0:
            return data  # No filtering if cutoff is too high
        b, a = butter(order, normal_cutoff, btype='high', analog=False)
        y = filtfilt(b, a, data)
        return y
    except Exception as e:
        logging.error(f"Error in high_pass_filter: {e}", exc_info=True)
        raise e

def apply_filter(data, type='high', cutoff=10, fs=100.0, order=5):
    """
    A general Butterworth filter (default high-pass).
    """
    try:
        nyquist = 0.5 * fs
        normal_cutoff = cutoff / nyquist
        if normal_cutoff >= 1.0:
            return data
        
        btype = 'high' if type == 'high' else 'low'
        b, a = butter(order, normal_cutoff, btype=btype, analog=False)
        return filtfilt(b, a, data)
    except Exception as e:
        logging.error(f"Error in apply_filter: {e}", exc_info=True)
        return data
