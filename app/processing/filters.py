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
        logging.error(f"Error in high_pass_filter: {e}")
        raise e
