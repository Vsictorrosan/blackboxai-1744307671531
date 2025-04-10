"""
Test script to verify ML model functionality
"""

from predict import AQIPredictor
import pandas as pd
import numpy as np
from datetime import datetime

def test_prediction():
    try:
        # Initialize predictor
        predictor = AQIPredictor()
        print("✓ Successfully loaded ML model")
        
        # Test data
        test_features = {
            'pm25': 50,
            'pm10': 75,
            'co': 5,
            'no2': 30,
            'temperature': 25,
            'humidity': 60,
            'wind_speed': 10,
            'day': datetime.now().day,
            'hour': datetime.now().hour,
            'month': datetime.now().month,
            'is_weekend': 1 if datetime.now().weekday() >= 5 else 0,
            'gas_pollutant_index': 45.0,
            'temperature_rolling_mean_3h': 24.0,
            'temperature_rolling_mean_6h': 23.5,
            'temperature_rolling_mean_12h': 23.0,
            'humidity_rolling_mean_3h': 58.0,
            'humidity_rolling_mean_6h': 57.0,
            'humidity_rolling_mean_12h': 56.0,
            'wind_speed_rolling_mean_3h': 9.5,
            'wind_speed_rolling_mean_6h': 9.0,
            'wind_speed_rolling_mean_12h': 8.5
        }
        
        try:
            # Create DataFrame
            df = pd.DataFrame([test_features])
            print("\nTest features shape:", df.shape)
            print("Test features columns:", sorted(df.columns.tolist()))
            
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
            
            # Try to extract feature names from error message
            error_msg = str(ve)
            if "Feature names unseen at fit time:" in error_msg:
                unseen = error_msg.split("Feature names unseen at fit time:")[1].split("Feature names seen at fit time")[0]
                print("\nUnseen features:", unseen.strip())
            
            if "Feature names seen at fit time yet now missing:" in error_msg:
                missing = error_msg.split("Feature names seen at fit time yet now missing:")[1]
                print("\nMissing features:", missing.strip())
            
            return False
            
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        return False

if __name__ == "__main__":
    print("Testing AQI Prediction Model...")
    success = test_prediction()
    print("\nTest completed:", "✓ Success" if success else "❌ Failed")
