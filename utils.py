import numpy as np
from scipy.signal import butter, filtfilt
from scipy.fft import fft
import pandas as pd
from sqlalchemy import create_engine, Column, Integer, String, Float, ForeignKey, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker
import os
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO, filename="app.log", format="%(asctime)s %(levelname)s:%(message)s"
)

Base = declarative_base()

class BallSize(Base):
    __tablename__ = 'ball_sizes'
    id = Column(Integer, primary_key=True)
    size = Column(String, unique=True, nullable=False)
    torque_min = Column(Float, nullable=False)
    torque_max = Column(Float, nullable=False)
    measurements = relationship("Measurement", back_populates="ball_size")

class Measurement(Base):
    __tablename__ = 'measurements'
    id = Column(Integer, primary_key=True)
    file_path = Column(String, unique=True, nullable=False)
    operator_id = Column(String, nullable=False)
    ball_size_id = Column(Integer, ForeignKey('ball_sizes.id'))
    timestamp = Column(DateTime, nullable=False)
    status = Column(String, default='Pending')  # e.g., Pending, Processed, Flagged
    ball_size = relationship("BallSize", back_populates="measurements")
    features = relationship("Feature", back_populates="measurement")

class Feature(Base):
    __tablename__ = 'features'
    id = Column(Integer, primary_key=True)
    measurement_id = Column(Integer, ForeignKey('measurements.id'))
    feature_name = Column(String, nullable=False)
    feature_value = Column(Float, nullable=False)
    measurement = relationship("Measurement", back_populates="features")

# Initialize Database
DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///torque_data.db')  # Replace with PostgreSQL URL in production
engine = create_engine(DATABASE_URL, echo=False)
Base.metadata.create_all(engine)
SessionLocal = sessionmaker(bind=engine)

# Define ALLOWED_DIRECTORY via environment variable or default
ALLOWED_DIRECTORY = os.getenv('ALLOWED_DIRECTORY', 'W:\\')  # Set to the desired directory

def high_pass_filter(data, cutoff=0.1, fs=100.0, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    if normal_cutoff >= 1.0:
        return data  # No filtering if cutoff is too high
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    y = filtfilt(b, a, data)
    return y

def calculate_fft(data, fs=100.0):
    N = len(data)
    T = 1.0 / fs
    yf = fft(data)
    xf = np.linspace(0.0, 1.0 / (2.0 * T), N // 2)
    amplitudes = 2.0 / N * np.abs(yf[:N // 2])
    return xf, amplitudes

def calculate_rms(series):
    return np.sqrt(np.mean(series**2))

def extract_all_features(data, cutoff=0.1, fs=100.0, order=5):
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
        logging.error(f"Error in feature extraction: {e}")
        return {}

def load_data(file_path):
    try:
        data = pd.read_csv(file_path, encoding='shift-jis')
        data.columns = [col.strip() for col in data.columns]
        return data
    except UnicodeDecodeError:
        # Attempt with 'utf-8' encoding
        try:
            data = pd.read_csv(file_path, encoding='utf-8')
            data.columns = [col.strip() for col in data.columns]
            return data
        except Exception as e:
            logging.error(f"Error loading data from {file_path}: {e}")
            raise e
    except Exception as e:
        logging.error(f"Error loading data from {file_path}: {e}")
        raise e

def is_safe_path(basedir, path):
    basedir = os.path.abspath(basedir)
    path = os.path.abspath(path)
    return os.path.commonpath([basedir, path]) == basedir

def detected_sudden_spike(filtered_rms, spike_threshold=0.1):
    diff = np.abs(np.diff(filtered_rms))
    spike_detected = np.any(diff > spike_threshold)
    return spike_detected

def analyse_hpf_rms(filtered_rms, threshold):
    average_rms = filtered_rms.mean()
    if average_rms > threshold:
        result = f"HPF_RMS is over the threshold ({threshold}): {average_rms:.4f}"
    else:
        result = f"HPF_RMS is within the threshold ({threshold}): {average_rms:.4f}."
    return result
