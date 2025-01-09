# app/tasks/measurement_tasks.py
from celery_worker import celery_app
from ..processing.data_loader import load_data
from ..processing.filters import high_pass_filter
from ..processing.feature_extraction import extract_all_features, calculate_fft, calculate_rms
from ..processing.anomaly_detection import detected_sudden_spike, analyse_hpf_rms
from ..plots.plot_factory import create_normal_plot, create_filtered_plot, create_fft_plot
from ..database.session import SessionLocal
from ..database.models import Measurement, Feature
import plotly.graph_objs as go
import pandas as pd
import logging
import os

@celery_app.task
def process_measurement(file_path, cutoff_freq, rms_window_size, hpf_rms_threshold, spike_threshold, y_axis_range, file_label, ball_size_id, operator_id):
    normal_fig = go.Figure()
    filtered_fig = go.Figure()
    fft_fig = go.Figure()
    features_text = ""
    analysis_result_text = ""

    from ..config import Config
    # Security check
    if not is_safe_path(Config.ALLOWED_DIRECTORY, file_path):
        logging.warning(f"Attempt to access invalid file path: {file_path}")
        error_fig = go.Figure()
        error_fig.add_annotation(
            text="Invalid file path selected.",
            xref="paper",
            yref="paper",
            showarrow=False,
            font=dict(color="red", size=16),
        )
        return [error_fig, go.Figure(), go.Figure()], "", ""
    
    try:
        # Load data
        data = load_data(file_path)
        x = data["X[mm]"]
        y = data["N[Ncm]"]
        
        # High-Pass Filtering
        y_filtered = high_pass_filter(y.to_numpy(), cutoff=cutoff_freq)
        filtered_series = pd.Series(y_filtered)
        filtered_rms = filtered_series.rolling(window=rms_window_size).apply(calculate_rms, raw=True)
        
        # Compute moving max and min of the filtered data
        moving_max = filtered_series.rolling(window=100, min_periods=1).max()
        moving_min = filtered_series.rolling(window=100, min_periods=1).min()
        
        # Compute moving average of the moving max and min
        moving_max_avg = moving_max.rolling(window=int(rms_window_size), min_periods=1).mean()
        moving_min_avg = moving_min.rolling(window=int(rms_window_size), min_periods=1).mean()
        
        # FFT
        xf, amplitudes = calculate_fft(y.to_numpy())
        
        # Create plots
        normal_fig = create_normal_plot(x, y, y_axis_range)
        filtered_fig = create_filtered_plot(
            x, y_filtered, filtered_rms, moving_max_avg, moving_min_avg, cutoff_freq, y_axis_range
        )
        fft_fig = create_fft_plot(xf, amplitudes)
        
        # Extract features
        features = extract_all_features(y, cutoff=cutoff_freq)
        features_text = "\n".join([f"{key}: {value}" for key, value in features.items()])
        
        # Generate analysis result
        analysis_result = []
        spike_detected = detected_sudden_spike(filtered_rms, spike_threshold)
        if spike_detected:
            analysis_result.append(
                "Sudden spike detected in filtered RMS data. Re-measurement recommended."
            )
        else:
            analysis_result.append("No sudden spike detected in filtered RMS data.")
        
        hpf_rms_result = analyse_hpf_rms(filtered_rms, hpf_rms_threshold)
        analysis_result.append(hpf_rms_result)
        
        if spike_detected or "over the threshold" in hpf_rms_result.lower():
            overall_result = "FAILED"
        else:
            overall_result = "PASSED"
        
        analysis_result_text = "\n".join(analysis_result)
        analysis_result_text = f"Overall analysis result: {overall_result}\n{analysis_result_text}"
        
        operator_id = operator_id if operator_id else "Unknown Operator"
        timestamp = pd.Timestamp.now().isoformat()
        features_text += f"\nOperator ID: {operator_id}\nTimestamp: {timestamp}"
        
        # Store measurement and features in the database
        session = SessionLocal()
        try:
            measurement = Measurement(
                file_path=file_path,
                operator_id=operator_id,
                ball_size_id=ball_size_id,
                timestamp=pd.Timestamp.now(),
                status='Processed'
            )
            session.add(measurement)
            session.flush()
            
            feature_objects = [
                Feature(
                    measurement_id=measurement.id,
                    feature_name=name,
                    feature_value=value
                )
                for name, value in features.items()
            ]
            session.bulk_save_objects(feature_objects)
            session.commit()
        except Exception as e:
            logging.error(f"Error storing measurement and features: {e}")
            session.rollback()
        finally:
            session.close()
        
        return [normal_fig, filtered_fig, fft_fig], features_text, analysis_result_text
    except Exception as e:
        logging.error(f"Error processing file {file_path}: {e}")
        error_fig = go.Figure()
        error_fig.add_annotation(
            text=f"Error processing data: {e}",
            xref="paper",
            yref="paper",
            showarrow=False,
            font=dict(color="red", size=16),
        )
        return [error_fig, go.Figure(), go.Figure()], "", ""
