"""
Test script to verify ML model functionality
"""

from predict import AQIPredictor
from aqi_test_features import get_test_features
import pandas as pd
from config import FEATURE_COLUMNS

def test_prediction():
    try:
        # Initialize predictor
        predictor = AQIPredictor()
        print("✓ Successfully loaded ML model")
        
        # Get test features
        test_features = get_test_features()
        
        try:
            # Create DataFrame to check feature alignment
            df = pd.DataFrame([test_features])
            
            print("\nExpected features:", sorted(FEATURE_COLUMNS))
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
            expected = set(FEATURE_COLUMNS)
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
