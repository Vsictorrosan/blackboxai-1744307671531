"""
Test features for AQI prediction model
"""

from datetime import datetime

def get_test_features():
    """Get test features that match the model's expected feature names"""
    now = datetime.now()
    
    return {
        # Basic measurements
        'pm25': 50.0,
        'pm10': 75.0,
        'gas_pollutant_index': 45.0,
        
        # Time features
        'hour': now.hour,
        'day': now.day,
        'month': now.month,
        
        # Weather measurements
        'temperature': 25.0,
        'humidity': 60.0,
        'wind_speed': 10.0,
        
        # Rolling averages for weather
        'temperature_rolling_mean_3h': 24.0,
        'temperature_rolling_mean_6h': 23.5,
        'temperature_rolling_mean_12h': 23.0,
        'humidity_rolling_mean_3h': 58.0,
        'humidity_rolling_mean_6h': 57.0,
        'humidity_rolling_mean_12h': 56.0,
        'wind_speed_rolling_mean_3h': 9.5,
        'wind_speed_rolling_mean_6h': 9.0,
        'wind_speed_rolling_mean_12h': 8.5,
        
        # Air quality measurements
        'co_level': 5.0,
        'no2_level': 30.0,
        'so2_level': 15.0,
        'o3_level': 40.0,
        
        # Traffic and industrial
        'traffic_density': 70.0,
        'industrial_activity': 50.0,
        'construction_intensity': 30.0,
        
        # Environmental conditions
        'precipitation': 0.0,
        'solar_radiation': 700.0,
        'atmospheric_pressure': 1013.0
    }
