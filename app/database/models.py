# app/database/models.py
from sqlalchemy import Column, Integer, String, Float, ForeignKey, DateTime, Text, UniqueConstraint
from sqlalchemy.orm import relationship
from .session import Base
import datetime

# 1) Define Feature first (or at least above Measurement)
class Feature(Base):
    __tablename__ = 'features'
    id = Column(Integer, primary_key=True)
    measurement_id = Column(Integer, ForeignKey('measurements.id'))
    feature_name = Column(String, nullable=False)
    feature_value = Column(Float, nullable=False)

    # relationship back to Measurement
    measurement = relationship("Measurement", back_populates="features")


class Measurement(Base):
    __tablename__ = 'measurements'
    id = Column(Integer, primary_key=True)

    # Basic fields
    file_path = Column(String, nullable=False)
    operator_id = Column(String, nullable=False)
    order_id = Column(String, nullable=True)  # New lot/order ID
    
    # Changed from ball_size_id (FK) to ball_size (Float)
    ball_size = Column(Float, nullable=True)
    
    measurement_time = Column(DateTime, nullable=False)
    analysis_time = Column(DateTime, nullable=False)
    submitted_timestamp = Column(DateTime, nullable=False)
    
    
    status = Column(String, default='Pending')

    label = Column(String, nullable=True)
    notes = Column(Text, nullable=True)
    
    # >>> NEW COLUMNS <<< about model prediction 
    predicted_label = Column(String, nullable=True)
    prediction_confidence = Column(Float, nullable=True)
    model_version = Column(String, nullable=True)
    
    # Add composite unique constraint
    __table_args__ = (
        UniqueConstraint('file_path', 'model_version', name='unique_file_model'),
    )


    # Relationship to BallSize REMOVED
    # ball_size = relationship("BallSize", back_populates="measurements")

    # Now the "Feature" class is already defined above
    features = relationship("Feature", back_populates="measurement")

    # Wide-table columns for features
    mean = Column(Float, nullable=True)
    median = Column(Float, nullable=True)
    mad = Column(Float, nullable=True)
    standard_deviation = Column(Float, nullable=True)
    rms = Column(Float, nullable=True)
    shape_factor = Column(Float, nullable=True)
    crest_factor = Column(Float, nullable=True)
    entropy = Column(Float, nullable=True)
    skewness = Column(Float, nullable=True)
    kurtosis = Column(Float, nullable=True)
    gradient_mean = Column(Float, nullable=True)
    gradient_std_dev = Column(Float, nullable=True)
    rolling_median_mean = Column(Float, nullable=True)
    rolling_mad_mean = Column(Float, nullable=True)
    spectral_centroid = Column(Float, nullable=True)
    spectral_entropy = Column(Float, nullable=True)
    peak_frequency = Column(Float, nullable=True)
    spectral_flatness = Column(Float, nullable=True)
    spectral_spread = Column(Float, nullable=True)
    spectral_roll_off = Column(Float, nullable=True)
    low_band_energy = Column(Float, nullable=True)
    mid_band_energy = Column(Float, nullable=True)
    high_band_energy = Column(Float, nullable=True)
    spectral_crest = Column(Float, nullable=True)
    spectral_flux = Column(Float, nullable=True)
    spectral_kurtosis = Column(Float, nullable=True)
    spectral_skewness = Column(Float, nullable=True)
    spectral_slope = Column(Float, nullable=True)

class AuditLog(Base):
    __tablename__ = 'audit_logs'
    id = Column(Integer, primary_key=True)
    measurement_id = Column(Integer, ForeignKey('measurements.id'), nullable=False)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    changed_by = Column(String, nullable=False) # Operator ID or System
    previous_status = Column(String)
    new_status = Column(String)
    previous_label = Column(String)
    new_label = Column(String)
    change_reason = Column(String) # For notes or automated reasons
    
    # Relationship
    measurement = relationship("Measurement")

class ModelRegistry(Base):
    __tablename__ = 'model_registry'
    id = Column(Integer, primary_key=True)
    version = Column(String, unique=True, nullable=False) # e.g. "v1.0.0_20241025" or UUID
    algorithm_type = Column(String, nullable=False) # "RandomForest", "LogisticRegression"
    hyperparameters = Column(Text) # JSON string
    test_accuracy = Column(Float)
    training_date = Column(DateTime, default=datetime.datetime.utcnow)
    file_path = Column(String, nullable=False) # Path to .pkl
    is_active = Column(Integer, default=1) # 1=Active, 0=Archived
    created_by = Column(String) # Admin/User ID
