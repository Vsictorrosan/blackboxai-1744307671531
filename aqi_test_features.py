"""
Test features for AQI prediction model
"""

from datetime import datetime

def get_test_features():
    """Get test features that match the model's expected feature names from config.py"""
    return {
        # Primary Features (High Importance > 10%)
        'pm25': 50.0,
        'pm10': 75.0,
        'emission_levels': 40.0,
        
        # Secondary Features (2-5%)
        'co': 5.0,
        'no2': 30.0,
        'daily_vehicle_count': 1000,
        'temperature': 25.0,
        'humidity': 60.0,
        
        # Industrial and Environmental Features
        'industrial_activity_index': 50.0,
        'industrial_consumption': 800.0,
        'power_demand': 1000.0,
        'production_index': 80.0,
        'energy_price_index': 100.0,
        
        # Urban and Safety Metrics
        'green_cover_percentage': 30.0,
        'urban_density': 50.0,
        'peak_hour_density': 70.0,
        'severity_index': 50.0,
        'compliance_score': 80.0,
        'violation_index': 20.0,
        
        # Environmental Conditions
        'pollen_level': 30.0,
        'wind_speed': 10.0,
        'season_type': 1,
        
        # Time Features
        'day_of_week': datetime.now().weekday(),
        'month': datetime.now().month
    }
