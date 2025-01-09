# app/database/models.py
from sqlalchemy import Column, Integer, String, Float, ForeignKey, DateTime
from sqlalchemy.orm import relationship
from .session import Base

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
