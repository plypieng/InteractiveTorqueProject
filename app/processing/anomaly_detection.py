# app/processing/anomaly_detection.py
import numpy as np
import logging
from sqlalchemy.orm import Session

def detected_sudden_spike(filtered_rms, spike_threshold=0.1):
    try:
        diff = np.abs(np.diff(filtered_rms))
        spike_detected = np.any(diff > spike_threshold)
        return spike_detected
    except Exception as e:
        logging.error(f"Error in detected_sudden_spike: {e}", exc_info=True)
        return False

def analyse_hpf_rms(filtered_rms, threshold):
    try:
        average_rms = filtered_rms.mean()
        if average_rms > threshold:
            return f"HPF_RMS is over the threshold ({threshold}): {average_rms:.4f}"
        else:
            return f"HPF_RMS is within the threshold ({threshold}): {average_rms:.4f}."
    except Exception as e:
        logging.error(f"Error in analyse_hpf_rms: {e}", exc_info=True)
        return "Error in HPF_RMS analysis."

def detect_anomalous_measurement(torque_values: np.ndarray, ball_size_val: float, session: Session):
    """
    Check torque_values vs. ball_size specs.
    Since BallSize table is removed, we currently skip this check or need new logic.
    """
    try:
        # Placeholder logic: if we had a formula for min/max based on ball_size_val, we'd use it here.
        # For now, we just return False (no anomaly) to avoid breaking the app.
        return False, "Anomaly detection skipped (BallSize table removed)."
    except Exception as e:
        logging.error(f"Error in detect_anomalous_measurement: {e}", exc_info=True)
        return False, "Error during anomaly detection."
