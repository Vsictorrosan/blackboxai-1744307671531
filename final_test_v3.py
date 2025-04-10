"""
Test script to verify ML model functionality
"""

from predict import AQIPredictor, MODEL_FEATURES
import pandas as pd
import numpy as np
from datetime import datetime

def test_prediction():
    try:
        # Initialize predictor
        predictor = AQIPredictor()
        print("✓ Successfully loaded ML model")
        
        # Get current time info
        now = datetime.now()
        
        # Test data with correct feature set
        test_features = {
            # Time features
            'day': now.day,
            'hour': now.hour,
            'month': now.month,
            'day_of_week': now.weekday(),
            'is_weekend': 1 if now.weekday() >= 5 else 0,
            
            # Primary pollutants
            'pm25': 50.0,
            'pm10': 75.0,
            'o3': 40.0,
            'co': 5.0,
            'no2': 30.0,
            'so2': 15.0,
            'gas_pollutant_index': (5.0 + 30.0) / 2,  # Average of CO and NO2
            
            # Derived air quality metrics
            'pm_ratio': 50.0 / 75.0,  # PM2.5/PM10 ratio
            
            # Rolling means for PM2.5 and PM10
            'pm25_rolling_mean_3h': 48.0,
            'pm25_rolling_mean_6h': 45.0,
            'pm10_rolling_mean_3h': 73.0,
            'pm10_rolling_mean_6h': 70.0,
            
            # Weather conditions
            'temperature': 25.0,
            'humidity': 60.0,
            'wind_speed': 10.0,
            'wind_direction': 180.0,  # South wind
            'wind_pressure': 1015.0,  # Slightly above standard pressure
            'precipitation': 0.0,
            'pressure': 1013.25,  # Standard atmospheric pressure in hPa
            
            # Derived weather metrics
            'temp_humidity': 25.0 * 60.0,  # Temperature-humidity interaction
            
            # Traffic conditions
            'traffic_level': 70.0,  # Scale 0-100
            'traffic_pollution_index': 65.0,  # Scale 0-100
            
            # Rolling means for weather conditions
            'temperature_rolling_mean_3h': 24.0,
            'temperature_rolling_mean_6h': 23.5,
            'humidity_rolling_mean_3h': 58.0,
            'humidity_rolling_mean_6h': 57.0
        }
        
        try:
            # Create DataFrame
            df = pd.DataFrame([test_features])
            
            print("\nExpected features:", sorted(MODEL_FEATURES))
            print("\nProvided features:", sorted(df.columns.tolist()))
            
            # Make prediction
            result = predictor.predict(test_features)
            print("\nPrediction Result:")
            print(f"AQI: {result['aqi']}")
            print(f"Risk Level: {result['risk_level']}")
            print(f"Confidence: {result['confidence']}")
            print(f"Contributing Factors: {result['factors']}")
            
            return True
            
        except ValueError as ve:
            print("\nFeature mismatch error:")
            print(str(ve))
            
            # Compare features
            expected = set(MODEL_FEATURES)
            provided = set(df.columns)
            
            print("\nMissing features:", sorted(expected - provided))
            print("Extra features:", sorted(provided - expected))
            
            return False
            
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        return False

if __name__ == "__main__":
    print("Testing AQI Prediction Model...")
    success = test_prediction()
    print("\nTest completed:", "✓ Success" if success else "❌ Failed")
