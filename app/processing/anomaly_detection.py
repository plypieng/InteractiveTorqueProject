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
        logging.error(f"Error in detected_sudden_spike: {e}")
        return False

def analyse_hpf_rms(filtered_rms, threshold):
    try:
        average_rms = filtered_rms.mean()
        if average_rms > threshold:
            result = f"HPF_RMS is over the threshold ({threshold}): {average_rms:.4f}"
        else:
            result = f"HPF_RMS is within the threshold ({threshold}): {average_rms:.4f}."
        return result
    except Exception as e:
        logging.error(f"Error in analyse_hpf_rms: {e}")
        return "Error in HPF_RMS analysis."

def detect_anomalous_measurement(torque_values: np.ndarray, ball_size_id: int, session: Session):
    """
    Detects anomalies in torque measurements based on ball size specifications.

    Parameters:
    - torque_values (np.ndarray): Array of torque measurements.
    - ball_size_id (int): ID of the ball size to retrieve torque specifications.
    - session (Session): SQLAlchemy session for database access.

    Returns:
    - anomaly_detected (bool): True if anomalies are found, False otherwise.
    - message (str): Description of the anomaly status.
    """
    try:
        # Retrieve BallSize from the database
        ball_size = session.query(BallSize).filter(BallSize.id == ball_size_id).first()
        if not ball_size:
            error_msg = f"BallSize with ID {ball_size_id} not found."
            logging.error(error_msg)
            return False, error_msg

        torque_min = ball_size.torque_min
        torque_max = ball_size.torque_max

        # Check if any torque values are outside the acceptable range
        anomalies = torque_values[(torque_values < torque_min) | (torque_values > torque_max)]
        if anomalies.size > 0:
            anomaly_detected = True
            anomaly_count = anomalies.size
            anomaly_details = f"{anomaly_count} torque measurements out of range [{torque_min}, {torque_max}]."
            message = f"Anomaly Detected: {anomaly_details}"
            logging.info(message)
            return anomaly_detected, message
        else:
            anomaly_detected = False
            message = "No anomalies detected in torque measurements."
            logging.info(message)
            return anomaly_detected, message

    except Exception as e:
        logging.error(f"Error in detect_anomalous_measurement: {e}")
        return False, "Error during anomaly detection."
