"""
API Integration Layer for Smart AQI Guardian
Handles multiple API calls and data aggregation
"""

import requests
import json
from datetime import datetime
from typing import Dict, Any, Optional
import logging
from functools import lru_cache
from time import time

class APIIntegrator:
    def __init__(self):
        self.logger = logging.getLogger('APIIntegrator')
        self.api_keys = {
            'openaq': '27c1d9c3e2eb419c2eea48b7f2a58f02732e1e3a66f6443a7f50cb4f3d342ee4',
            'weatherapi': '6c2906538a8748a2a8955531251004',
            'openweather': 'fe28e65d98738dbf772ee4901fbc35e3',
            'waqi': '9a6e23f0c8286868d87225a2f3819be997cfff59',
            'tomtom': 'IDrXu1GaubPGJIHrGbmav6hbGaiuQM6D',
            'geoapify': '421948bf03514a878ecf3a93b7b34ffc',
            'stlouisfed': 'df3cf23f3fa98f8e1def72ee02ccd085',
            'opencage': 'f618b3db4e9a4d77b76a5248bd51e821'
        }
        
    @lru_cache(maxsize=128)
    def _cached_request(self, url: str, headers: Optional[Dict] = None) -> Dict:
        """Make a cached API request."""
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            self.logger.error(f"API request failed: {str(e)}")
            return {}

    def get_air_quality(self, city: str) -> Dict[str, Any]:
        """Get air quality data from multiple sources."""
        try:
            # OpenAQ API
            openaq_headers = {'X-API-Key': self.api_keys['openaq']}
            openaq_url = f"https://api.openaq.org/v3/locations?city={city}&limit=1"
            openaq_data = self._cached_request(openaq_url, headers=openaq_headers)

            # WAQI API
            waqi_url = f"https://api.waqi.info/feed/{city}/?token={self.api_keys['waqi']}"
            waqi_data = self._cached_request(waqi_url)

            return {
                'openaq': openaq_data,
                'waqi': waqi_data,
                'timestamp': datetime.utcnow().isoformat()
            }
        except Exception as e:
            self.logger.error(f"Error getting air quality data: {str(e)}")
            return {}

    def get_weather(self, city: str) -> Dict[str, Any]:
        """Get weather data from multiple sources."""
        try:
            # WeatherAPI
            weather_url = f"http://api.weatherapi.com/v1/current.json?key={self.api_keys['weatherapi']}&q={city}"
            weather_data = self._cached_request(weather_url)

            # OpenWeatherMap
            openweather_url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={self.api_keys['openweather']}&units=metric"
            openweather_data = self._cached_request(openweather_url)

            return {
                'weatherapi': weather_data,
                'openweather': openweather_data,
                'timestamp': datetime.utcnow().isoformat()
            }
        except Exception as e:
            self.logger.error(f"Error getting weather data: {str(e)}")
            return {}

    def get_traffic(self, lat: float, lon: float) -> Dict[str, Any]:
        """Get traffic data from multiple sources."""
        try:
            # TomTom Traffic
            tomtom_url = f"https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json?point={lat},{lon}&key={self.api_keys['tomtom']}"
            tomtom_data = self._cached_request(tomtom_url)

            # Geoapify
            geoapify_url = f"https://api.geoapify.com/v1/routing?waypoints={lat},{lon}|{lat+0.1},{lon+0.1}&mode=drive&traffic=approximated&apiKey={self.api_keys['geoapify']}"
            geoapify_data = self._cached_request(geoapify_url)

            return {
                'tomtom': tomtom_data,
                'geoapify': geoapify_data,
                'timestamp': datetime.utcnow().isoformat()
            }
        except Exception as e:
            self.logger.error(f"Error getting traffic data: {str(e)}")
            return {}

    def get_industrial_activity(self) -> Dict[str, Any]:
        """Get industrial activity data."""
        try:
            # St. Louis FED Industrial Production
            fred_url = f"https://api.stlouisfed.org/fred/series/observations?series_id=INDPRO&api_key={self.api_keys['stlouisfed']}&file_type=json"
            fred_data = self._cached_request(fred_url)

            return {
                'industrial_production': fred_data,
                'timestamp': datetime.utcnow().isoformat()
            }
        except Exception as e:
            self.logger.error(f"Error getting industrial data: {str(e)}")
            return {}

    def get_urban_development(self, location: str) -> Dict[str, Any]:
        """Get urban development data."""
        try:
            # OpenCage Geocoding
            opencage_url = f"https://api.opencagedata.com/geocode/v1/json?q={location}&key={self.api_keys['opencage']}"
            opencage_data = self._cached_request(opencage_url)

            return {
                'urban_data': opencage_data,
                'timestamp': datetime.utcnow().isoformat()
            }
        except Exception as e:
            self.logger.error(f"Error getting urban data: {str(e)}")
            return {}

    def get_aggregated_data(self, city: str, lat: float, lon: float) -> Dict[str, Any]:
        """Get aggregated data from all sources."""
        return {
            'air_quality': self.get_air_quality(city),
            'weather': self.get_weather(city),
            'traffic': self.get_traffic(lat, lon),
            'industrial': self.get_industrial_activity(),
            'urban': self.get_urban_development(city),
            'timestamp': datetime.utcnow().isoformat()
        }
