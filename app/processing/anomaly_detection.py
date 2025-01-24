# app/processing/anomaly_detection.py
import numpy as np
import logging
from sqlalchemy.orm import Session
from ..database.models import BallSize

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

def detect_anomalous_measurement(torque_values: np.ndarray, ball_size_id: int, session: Session):
    """
    If you want to check torque_values vs. ball_size specs in DB
    """
    try:
        ball_size = session.query(BallSize).filter(BallSize.id == ball_size_id).first()
        if not ball_size:
            error_msg = f"BallSize with ID {ball_size_id} not found."
            logging.error(error_msg)
            return False, error_msg

        torque_min = ball_size.torque_min
        torque_max = ball_size.torque_max
        anomalies = torque_values[(torque_values < torque_min) | (torque_values > torque_max)]
        if anomalies.size > 0:
            anomaly_count = anomalies.size
            message = f"Anomaly Detected: {anomaly_count} out of range [{torque_min}, {torque_max}]."
            logging.info(message)
            return True, message
        else:
            return False, "No anomalies detected in torque measurements."
    except Exception as e:
        logging.error(f"Error in detect_anomalous_measurement: {e}", exc_info=True)
        return False, "Error during anomaly detection."
