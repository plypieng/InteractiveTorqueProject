# app/processing/measurement_processor.py
"""
Synchronous measurement processing module.
Centralized logic for torque data analysis.
"""
from .data_loader import load_data
from .filters import high_pass_filter
from .feature_extraction import extract_features, calculate_fft, calculate_rms
from .anomaly_detection import detected_sudden_spike, analyse_hpf_rms
from ..plots.plot_factory import create_normal_plot, create_filtered_plot, create_fft_plot
from ..database.session import SessionLocal
from ..database.models import Measurement, Feature
import plotly.graph_objs as go
import pandas as pd
import numpy as np
import logging
import os
import datetime

def analyze_torque_data(file_path, cutoff_freq, rms_window_size, hpf_rms_threshold, spike_threshold, y_axis_range, ball_size_id, operator_id, file_path_2=None):
    """
    Core logic for processing torque data, generating plots, and extracting features.
    Supports optional comparison with a second file.
    """
    from ..config import Config
    from ..utils.file_security import is_safe_path
    
    # Initialize return values
    figs = [go.Figure(), go.Figure(), go.Figure()]
    features_dict = {}
    analysis_result_text = ""

    if not is_safe_path(Config.ALLOWED_DIRECTORY, file_path):
        error_msg = "Invalid file path selected."
        logging.warning(f"Attempt to access invalid file path: {file_path}")
        figs[0].add_annotation(text=error_msg, xref="paper", yref="paper", showarrow=False, font=dict(color="red", size=16))
        return figs, features_dict, error_msg

    try:
        # Load File 1
        data1 = load_data(file_path)
        x1 = data1["X[mm]"]
        y1 = data1["N[Ncm]"]
        
        # Extract features for File 1
        features_dict = extract_features(y1.to_numpy(), fs=100.0, window_size=rms_window_size)
        
        # Process File 1
        y1_filtered = high_pass_filter(y1.to_numpy(), cutoff=cutoff_freq)
        f1_series = pd.Series(y1_filtered)
        f1_rms = f1_series.rolling(window=rms_window_size).apply(calculate_rms, raw=True)
        f1_max_avg = f1_series.rolling(window=100, min_periods=1).max().rolling(window=int(rms_window_size), min_periods=1).mean()
        f1_min_avg = f1_series.rolling(window=100, min_periods=1).min().rolling(window=int(rms_window_size), min_periods=1).mean()
        xf1, amp1 = calculate_fft(y1.to_numpy(), fs=100.0)

        # Handle File 2 (optional)
        x2, y2_filt, xf2, amp2 = None, None, None, None
        name2 = "File 2"
        if file_path_2 and is_safe_path(Config.ALLOWED_DIRECTORY, file_path_2):
            try:
                data2 = load_data(file_path_2)
                x2 = data2["X[mm]"]
                y2 = data2["N[Ncm]"]
                y2_filt = high_pass_filter(y2.to_numpy(), cutoff=cutoff_freq)
                xf2, amp2 = calculate_fft(y2.to_numpy(), fs=100.0)
                name2 = os.path.basename(file_path_2)
            except Exception as e:
                logging.error(f"Error loading second file for comparison: {e}")

        # Create comparative figures
        name1 = os.path.basename(file_path)
        normal_fig = create_normal_plot(x1, y1, y_axis_range, x2, (y2 if x2 is not None else None), name1, name2)
        filtered_fig = create_filtered_plot(x1, y1_filtered, f1_rms, f1_max_avg, f1_min_avg, cutoff_freq, [-1,1], x2, y2_filt)
        fft_fig = create_fft_plot(xf1, amp1, xf2, amp2)
        figs = [normal_fig, filtered_fig, fft_fig]

        # Anomaly Detection for File 1
        analysis_list = []
        spike_detected = detected_sudden_spike(f1_rms, spike_threshold)
        analysis_list.append("Sudden spike detected in F1 RMS data. Re-measurement recommended." if spike_detected else "No sudden spike detected in F1 RMS data.")
        
        hpf_rms_result = analyse_hpf_rms(f1_rms, hpf_rms_threshold)
        analysis_list.append(f"F1 HPF RMS: {hpf_rms_result}")
        
        overall_result = "FAILED" if (spike_detected or "over the threshold" in hpf_rms_result.lower()) else "PASSED"
        analysis_result_text = f"Overall analysis result: {overall_result}\n" + "\n".join(analysis_list)

        # Meta-data for features
        features_dict.update({
            "Analysis Date": pd.Timestamp.now().isoformat(),
            "Operator": operator_id if operator_id else "Unknown",
            "Ball Size": ball_size_id,
        })

        # Filename metadata
        parts = name1.split("_")
        if len(parts) >= 4:
            features_dict.update({
                "Measurement date": parts[0],
                "Size": parts[1],
                "Number": parts[2],
                "RPM": parts[3].split(".")[0]
            })

    except Exception as e:
        logging.error(f"Error analyzing torque data: {e}", exc_info=True)
        figs[0].add_annotation(text=f"Error: {e}", xref="paper", yref="paper", showarrow=False)
        analysis_result_text = f"Error: {e}"

    return figs, features_dict, analysis_result_text

def process_and_save_measurement(file_path, cutoff_freq, rms_window_size, hpf_rms_threshold, spike_threshold, y_axis_range, ball_size_id, operator_id):
    """
    Process measurement and save results to the database.
    """
    figs, features_dict, result_text = analyze_torque_data(
        file_path, cutoff_freq, rms_window_size, hpf_rms_threshold, spike_threshold, y_axis_range, ball_size_id, operator_id
    )
    
    # Store measurement + features in DB
    session = SessionLocal()
    try:
        measurement = Measurement(
            file_path=file_path,
            operator_id=operator_id,
            ball_size_id=ball_size_id,
            status='Processed'
        )
        session.add(measurement)
        session.flush()
        
        for name, value in features_dict.items():
            # Only save numeric features to the Feature table if appropriate
            if isinstance(value, (int, float, np.number)):
                row_feature = Feature(
                    measurement_id=measurement.id,
                    feature_name=name,
                    feature_value=float(value)
                )
                session.add(row_feature)

        session.commit()
    except Exception as e:
        logging.error(f"Error storing measurement to DB: {e}", exc_info=True)
        session.rollback()
    finally:
        session.close()
    
    return figs, features_dict, result_text

# Backward compatibility
def process_measurement(*args, **kwargs):
    return process_and_save_measurement(*args, **kwargs)
