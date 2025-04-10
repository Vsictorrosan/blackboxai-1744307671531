"""
Backend API for Smart AQI Guardian
Handles ML model predictions and serves them to the frontend
Integrates multiple data sources for comprehensive analysis
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
from predict import AQIPredictor
from datetime import datetime
from api_integrator import APIIntegrator

# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={
    r"/api/*": {"origins": ["http://localhost:3000"]},
    r"/predict": {"origins": ["http://localhost:3000"]}
})

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize components
try:
    predictor = AQIPredictor()
    api_integrator = APIIntegrator()
    logger.info("ML model and API integrator loaded successfully")
except Exception as e:
    logger.error(f"Initialization error: {str(e)}")
    raise

# New routes for API data
@app.route('/api/weather/<city>', methods=['GET'])
def get_weather(city):
    try:
        weather_data = api_integrator.get_weather(city)
        return jsonify(weather_data)
    except Exception as e:
        logger.error(f"Weather API error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/air-quality/<city>', methods=['GET'])
def get_air_quality(city):
    try:
        aqi_data = api_integrator.get_air_quality(city)
        return jsonify(aqi_data)
    except Exception as e:
        logger.error(f"AQI API error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/traffic', methods=['GET'])
def get_traffic():
    try:
        lat = float(request.args.get('lat', '28.6139'))
        lon = float(request.args.get('lon', '77.2090'))
        traffic_data = api_integrator.get_traffic(lat, lon)
        return jsonify(traffic_data)
    except Exception as e:
        logger.error(f"Traffic API error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/industrial', methods=['GET'])
def get_industrial():
    try:
        industrial_data = api_integrator.get_industrial_activity()
        return jsonify(industrial_data)
    except Exception as e:
        logger.error(f"Industrial API error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/urban/<location>', methods=['GET'])
def get_urban(location):
    try:
        urban_data = api_integrator.get_urban_development(location)
        return jsonify(urban_data)
    except Exception as e:
        logger.error(f"Urban API error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/aggregated/<city>', methods=['GET'])
def get_aggregated(city):
    try:
        lat = float(request.args.get('lat', '28.6139'))
        lon = float(request.args.get('lon', '77.2090'))
        data = api_integrator.get_aggregated_data(city, lat, lon)
        return jsonify(data)
    except Exception as e:
        logger.error(f"Aggregated API error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get features from request
        features = request.json
        city = features.get('city', 'Delhi')
        lat = float(features.get('lat', '28.6139'))
        lon = float(features.get('lon', '77.2090'))
        
        # Get real-time data from APIs
        api_data = api_integrator.get_aggregated_data(city, lat, lon)
        
        # Extract weather data
        weather = api_data['weather'].get('openweather', {}).get('main', {})
        
        # Extract air quality data
        air = api_data['air_quality'].get('waqi', {}).get('data', {})
        
        # Extract traffic data
        traffic = api_data['traffic'].get('tomtom', {})
        
        # Extract industrial data
        industrial = api_data['industrial'].get('industrial_production', {})
        
        # Combine API data with user input
        enhanced_features = {
            # Air Quality Parameters
            'pm25': float(air.get('iaqi', {}).get('pm25', {}).get('v', 0)),
            'pm10': float(air.get('iaqi', {}).get('pm10', {}).get('v', 0)),
            'co': float(air.get('iaqi', {}).get('co', {}).get('v', 0)),
            'no2': float(air.get('iaqi', {}).get('no2', {}).get('v', 0)),
            
            # Weather Parameters
            'temperature': float(weather.get('temp', 25)),
            'humidity': float(weather.get('humidity', 50)),
            'wind_speed': float(weather.get('wind_speed', 10)),
            
            # Time-based Parameters
            'season_type': features.get('season_type', 1),
            'day_of_week': datetime.now().weekday(),
            'month': datetime.now().month,
            
            # Urban Parameters
            'green_cover_percentage': features.get('green_cover_percentage', 30),
            'urban_density': features.get('urban_density', 50),
            
            # Traffic Parameters
            'daily_vehicle_count': float(traffic.get('flowSegmentData', {}).get('vehicleCount', 1000)),
            'peak_hour_density': features.get('peak_hour_density', 70),
            
            # Industrial Parameters
            'emission_levels': features.get('emission_levels', 50),
            'industrial_activity_index': float(industrial.get('value', 50)),
            'power_demand': features.get('power_demand', 1000),
            'production_index': features.get('production_index', 80),
            'energy_price_index': features.get('energy_price_index', 100),
            'industrial_consumption': features.get('industrial_consumption', 800),
            
            # Safety Parameters
            'severity_index': features.get('severity_index', 50),
            'compliance_score': features.get('compliance_score', 80),
            'violation_index': features.get('violation_index', 20),
            'pollen_level': features.get('pollen_level', 30)
        }
        
        # Make prediction with enhanced features
        result = predictor.predict(enhanced_features)
        
        # Add API data sources to result
        result['data_sources'] = {
            'weather': bool(weather),
            'air_quality': bool(air),
            'traffic': bool(traffic),
            'industrial': bool(industrial)
        }
        
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({
            'error': 'Prediction failed',
            'message': str(e)
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
