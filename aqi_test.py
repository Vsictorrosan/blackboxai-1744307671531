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
            'is_weekend': 1 if now.weekday() >= 5 else 0,
            
            # Air quality measurements
            'pm25': 50.0,
            'pm10': 75.0,
            'co': 5.0,
            'no2': 30.0,
            'gas_pollutant_index': (5.0 + 30.0) / 2,  # Average of CO and NO2
            
            # Current weather conditions
            'temperature': 25.0,
            'humidity': 60.0,
            'wind_speed': 10.0,
            
            # Rolling means for temperature
            'temperature_rolling_mean_3h': 24.0,
            'temperature_rolling_mean_6h': 23.5,
            'temperature_rolling_mean_12h': 23.0,
            
            # Rolling means for humidity
            'humidity_rolling_mean_3h': 58.0,
            'humidity_rolling_mean_6h': 57.0,
            'humidity_rolling_mean_12h': 56.0,
            
            # Rolling means for wind speed
            'wind_speed_rolling_mean_3h': 9.5,
            'wind_speed_rolling_mean_6h': 9.0,
            'wind_speed_rolling_mean_12h': 8.5
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
