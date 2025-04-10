"""
Test script to verify ML model functionality
"""

from predict import AQIPredictor

def test_prediction():
    try:
        # Initialize predictor
        predictor = AQIPredictor()
        print("✓ Successfully loaded ML model")
        
        # Test data
        test_features = {
            'pm25': 50,
            'pm10': 75,
            'emission_levels': 40,
            'co': 5,
            'no2': 30,
            'temperature': 25,
            'humidity': 60,
            'industrial_activity_index': 50,
            'daily_vehicle_count': 1000,
            'wind_speed': 10,
            'season_type': 1,
            'day_of_week': 3,
            'month': 6,
            'green_cover_percentage': 30,
            'urban_density': 50,
            'peak_hour_density': 70,
            'severity_index': 50,
            'compliance_score': 80,
            'violation_index': 20,
            'pollen_level': 30,
            'power_demand': 1000,
            'production_index': 80,
            'energy_price_index': 100,
            'industrial_consumption': 800
        }
        
        # Make prediction
        result = predictor.predict(test_features)
        print("\nPrediction Result:")
        print(f"AQI: {result['aqi']}")
        print(f"Risk Level: {result['risk_level']}")
        print(f"Confidence: {result['confidence']}")
        print(f"Contributing Factors: {result['factors']}")
        
        return True
    
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False

if __name__ == "__main__":
    print("Testing AQI Prediction Model...")
    success = test_prediction()
    print("\nTest completed:", "✓ Success" if success else "❌ Failed")
