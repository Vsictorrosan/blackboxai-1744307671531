"""
Configuration settings for AQI prediction model.
Project: smart-aqi-guardian
Author: Mithileshvinayak
Last Updated: 2025-04-10 11:35:24
"""

import os
from pathlib import Path

# Base directories
ML_MODEL_DIR = Path(__file__).parent
PROJECT_ROOT = ML_MODEL_DIR.parent

# Directory paths
MODEL_DIR = os.path.join(ML_MODEL_DIR, 'model')
DATA_DIR = os.path.join(PROJECT_ROOT, 'data_pipeline', 'processed')

# File paths
MODEL_FILENAME = 'aqi_model.joblib'
SCALER_FILENAME = 'scaler.joblib'
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)
SCALER_PATH = os.path.join(MODEL_DIR, SCALER_FILENAME)
TRAINING_DATA_PATH = os.path.join(DATA_DIR, 'aqi_data.csv')

# Create directories if they don't exist
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# Enhanced Model parameters for better performance
MODEL_CONFIG = {
    'n_estimators': 150,
    'max_depth': 15,
    'min_samples_split': 4,
    'min_samples_leaf': 2,
    'n_jobs': -1,
    'random_state': 42
}

# Updated Feature configuration
FEATURE_COLUMNS = [
    # Primary Features (High Importance > 10%)
    'pm25',
    'pm10',
    'emission_levels',
    
    # Secondary Features (2-5%)
    'co',
    'no2',
    'daily_vehicle_count',
    'temperature',
    'humidity',
    
    # Industrial and Environmental Features
    'industrial_activity_index',
    'industrial_consumption',
    'power_demand',
    'production_index',
    'energy_price_index',
    
    # Urban and Safety Metrics
    'green_cover_percentage',
    'urban_density',
    'peak_hour_density',
    'severity_index',
    'compliance_score',
    'violation_index',
    
    # Environmental Conditions
    'pollen_level',
    'wind_speed',
    'season_type',
    
    # Time Features
    'day_of_week',
    'month'
]

TARGET_COLUMN = 'aqi'

# Training parameters
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Logging configuration
LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'
LOG_LEVEL = 'INFO'