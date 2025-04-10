"""
Training script for AQI prediction model.
Project: smart-aqi-guardian
Author: Mithileshvinayak
Last Updated: 2025-04-10 11:35:24
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
from datetime import datetime
from pathlib import Path

# Add project root to Python path
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.append(str(project_root))

# Import config
from ml_model.config import *

# Setup logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format=LOG_FORMAT
)

class AQIModel:
    def __init__(self):
        """Initialize the AQI prediction model."""
        self.model = RandomForestRegressor(**MODEL_CONFIG)
        self.scaler = StandardScaler()
        self.features = FEATURE_COLUMNS
        self.target = TARGET_COLUMN
        self.logger = logging.getLogger('AQIModel')

    def preprocess_data(self, df):
        """Preprocess the input data."""
        self.logger.info("Preprocessing data...")
        
        df = df.copy()
        
        # Extract datetime features if timestamp column exists
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['day_of_week'] = df['timestamp'].dt.dayofweek
            df['month'] = df['timestamp'].dt.month
            self.logger.info("Extracted time features from timestamp")
        else:
            self.logger.warning("No timestamp column found. Creating default time features.")
            current_time = datetime.now()
            df['day_of_week'] = current_time.weekday()
            df['month'] = current_time.month

        # Handle categorical variables
        season_mapping = {
            'winter': 0,
            'spring': 1,
            'summer': 2,
            'fall': 3,
            'autumn': 3  # Alternative name for fall
        }
        
        # Convert season_type to lowercase and map to numbers
        if 'season_type' in df.columns:
            df['season_type'] = df['season_type'].str.lower().map(season_mapping)
            self.logger.info("Encoded season_type to numerical values")
        
        # Add missing columns with default values
        default_values = {
            # Primary Air Quality Parameters
            'pm25': (50.0, 'µg/m³'),
            'pm10': (100.0, 'µg/m³'),
            'no2': (40.0, 'µg/m³'),
            'co': (4.4, 'mg/m³'),
            
            # Weather Parameters
            'temperature': (25.0, '°C'),
            'humidity': (60.0, '%'),
            'wind_speed': (10.0, 'm/s'),
            
            # Traffic and Urban Parameters
            'daily_vehicle_count': (5000, 'vehicles/day'),
            'peak_hour_density': (70.0, '%'),
            'urban_density': (60.0, '%'),
            
            # Industrial Parameters
            'emission_levels': (50.0, 'index'),
            'industrial_activity_index': (45.0, 'index'),
            'industrial_consumption': (55.0, 'MW'),
            'power_demand': (75.0, 'MW'),
            'production_index': (65.0, 'index'),
            'energy_price_index': (80.0, 'index'),
            
            # Environmental Parameters
            'green_cover_percentage': (25.0, '%'),
            'pollen_level': (30.0, 'index'),
            'season_type': (2, 'category'),  # Summer as default
            
            # Safety and Compliance
            'severity_index': (40.0, 'index'),
            'compliance_score': (85.0, 'score'),
            'violation_index': (15.0, 'index')
        }
        
        for col in self.features:
            if col not in df.columns:
                if col in default_values:
                    default_value, unit = default_values[col]
                    self.logger.warning(f"Adding default {col} column ({default_value} {unit})")
                    df[col] = default_value
                else:
                    self.logger.warning(f"Column {col} not found and no default value available")
        
        # Handle missing values with column-specific logic
        for col in self.features:
            if df[col].isnull().any():
                if col == 'season_type':
                    df[col].fillna(2, inplace=True)  # Use summer (2) as default
                else:
                    df[col].fillna(df[col].mean(), inplace=True)  # Use mean for numerical
        
        # Ensure all data is numeric
        for col in self.features:
            if not pd.api.types.is_numeric_dtype(df[col]):
                self.logger.error(f"Column {col} contains non-numeric data: {df[col].unique()}")
                raise ValueError(f"Column {col} must be numeric")
        
        # Scale features
        df[self.features] = self.scaler.fit_transform(df[self.features])
        
        self.logger.info("Data preprocessing completed successfully")
        return df

    def evaluate_model(self, X, y):
        """Evaluate model using cross-validation."""
        self.logger.info("Performing cross-validation...")
        cv_scores = cross_val_score(
            self.model, X, y, 
            cv=5, 
            scoring='neg_root_mean_squared_error'
        )
        return -cv_scores.mean(), cv_scores.std()

    def train(self, data_path):
        """Train the model with data from given path."""
        try:
            # Load and preprocess data
            self.logger.info(f"Loading data from {data_path}")
            df = pd.read_csv(data_path)
            
            # Check target column
            if self.target not in df.columns:
                if 'aqi' in df.columns:
                    self.logger.info(f"Using 'aqi' as target column instead of {self.target}")
                    self.target = 'aqi'
                else:
                    raise ValueError(f"Neither '{self.target}' nor 'aqi' found in dataset")
            
            # Validate data types before preprocessing
            for col in df.columns:
                if col in self.features:
                    self.logger.info(f"Column {col} type: {df[col].dtype}")
                    if not pd.api.types.is_numeric_dtype(df[col]) and col != 'season_type':
                        self.logger.error(f"Non-numeric data in column {col}: {df[col].unique()}")
            
            df = self.preprocess_data(df)
            
            # Additional validation after preprocessing
            if not df[self.features].select_dtypes(include=[np.number]).shape[1] == len(self.features):
                non_numeric = [col for col in self.features if not pd.api.types.is_numeric_dtype(df[col])]
                raise ValueError(f"Non-numeric columns after preprocessing: {non_numeric}")
            
            # Prepare features and target
            X = df[self.features]
            y = df[self.target]
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, 
                test_size=TEST_SIZE, 
                random_state=RANDOM_STATE
            )
            
            # Train model
            self.logger.info("Training model...")
            self.model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = self.model.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Cross-validation score
            cv_rmse, cv_std = self.evaluate_model(X, y)
            
            # Feature importance
            importance = dict(zip(self.features, 
                                self.model.feature_importances_))
            
            # Log results
            self.logger.info("\nModel Performance:")
            self.logger.info(f"RMSE: {rmse:.2f}")
            self.logger.info(f"MAE: {mae:.2f}")
            self.logger.info(f"R2 Score: {r2:.2f}")
            self.logger.info(f"Cross-val RMSE: {cv_rmse:.2f} (+/- {cv_std:.2f})")
            self.logger.info("\nFeature Importance:")
            for feat, imp in sorted(importance.items(), 
                                  key=lambda x: x[1], reverse=True):
                self.logger.info(f"{feat}: {imp:.4f}")
            
            # Save model and scaler
            self.save_model()
            
            return {
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'cv_rmse': cv_rmse,
                'cv_std': cv_std,
                'importance': importance
            }
            
        except Exception as e:
            self.logger.error(f"Error in training: {str(e)}")
            raise

    def save_model(self):
        """Save the trained model and scaler."""
        try:
            # Create model directory if it doesn't exist
            os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
            
            # Save model and scaler
            joblib.dump(self.model, MODEL_PATH)
            joblib.dump(self.scaler, SCALER_PATH)
            
            self.logger.info(f"Model saved to {MODEL_PATH}")
            self.logger.info(f"Scaler saved to {SCALER_PATH}")
            
        except Exception as e:
            self.logger.error(f"Error saving model: {str(e)}")
            raise

def main():
    """Main training function."""
    try:
        print(f"\n=== Starting AQI Model Training ===")
        print(f"Time: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"User: Mithileshvinayak\n")
        
        # Allow user to specify data path or use default
        data_path = input(f"Enter path to enhanced dataset (press Enter to use default {TRAINING_DATA_PATH}): ")
        data_path = data_path.strip() or TRAINING_DATA_PATH
        
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Dataset not found at: {data_path}")
        
        # Load and show data structure
        print("\nLoading dataset...")
        df = pd.read_csv(data_path)
        print("\nCurrent columns in dataset:")
        for col in df.columns:
            print(f"- {col}")
        
        print("\nRequired columns:")
        for col in FEATURE_COLUMNS:
            status = "✓" if col in df.columns else "✗"
            print(f"- {col}: {status}")
            
        # Show target column information
        target_status = "✓" if 'future_aqi' in df.columns else ("✓ (using 'aqi')" if 'aqi' in df.columns else "✗")
        print(f"\nTarget column (future_aqi): {target_status}")
            
        proceed = input("\nProceed with training using default values for missing columns? (y/n): ")
        if proceed.lower() != 'y':
            print("Training cancelled.")
            return
        
        model = AQIModel()
        metrics = model.train(data_path)
        
        print("\n=== Training Complete ===")
        print(f"RMSE: {metrics['rmse']:.2f}")
        print(f"MAE: {metrics['mae']:.2f}")
        print(f"R2 Score: {metrics['r2']:.2f}")
        print(f"Cross-val RMSE: {metrics['cv_rmse']:.2f} "
              f"(+/- {metrics['cv_std']:.2f})")
        
        print("\nTop 5 Important Features:")
        top_features = dict(sorted(
            metrics['importance'].items(),
            key=lambda x: x[1],
            reverse=True
        )[:5])
        for feature, importance in top_features.items():
            print(f"- {feature}: {importance:.4f}")
        
        print(f"\nModel and scaler saved in: {os.path.dirname(MODEL_PATH)}")
        
    except Exception as e:
        logging.error(f"Training failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()