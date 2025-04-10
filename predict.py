"""
AQI Prediction Module
Project: Smart AQI Guardian
Author: Mithileshvinayak
Last Updated: 2025-04-10 13:10:15
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List
import logging

from config import MODEL_PATH, SCALER_PATH

# Features expected by the trained model
MODEL_FEATURES = [
    # Time features
    'day', 'hour', 'month', 'day_of_week', 'is_weekend',
    
    # Primary pollutants
    'pm25', 'pm10', 'o3', 'co', 'no2', 'so2',
    'gas_pollutant_index',
    
    # Derived air quality metrics
    'pm_ratio',  # PM2.5/PM10 ratio
    
    # Rolling means for PM2.5 and PM10
    'pm25_rolling_mean_3h', 'pm25_rolling_mean_6h',
    'pm10_rolling_mean_3h', 'pm10_rolling_mean_6h',
    
    # Weather conditions
    'temperature', 'humidity', 'wind_speed',
    'precipitation', 'pressure',
    'wind_direction', 'wind_pressure',
    'temp_humidity',  # Temperature-humidity interaction
    
    # Traffic conditions
    'traffic_level',
    'traffic_pollution_index',
    
    # Rolling means for weather conditions
    'temperature_rolling_mean_3h', 'temperature_rolling_mean_6h',
    'humidity_rolling_mean_3h', 'humidity_rolling_mean_6h'
]

class AQIPredictor:
    def __init__(self):
        self.logger = logging.getLogger('AQIPredictor')
        self.model = self._load_model()
        self.scaler = self._load_scaler()
        self.features = MODEL_FEATURES

    def _load_model(self):
        """Load the trained model."""
        try:
            return joblib.load(MODEL_PATH)
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            raise

    def _load_scaler(self):
        """Load the fitted scaler."""
        try:
            return joblib.load(SCALER_PATH)
        except Exception as e:
            self.logger.error(f"Error loading scaler: {str(e)}")
            raise

    def _get_risk_level(self, aqi: float) -> str:
        """Determine risk level based on AQI value."""
        if aqi <= 50:
            return "Good"
        elif aqi <= 100:
            return "Moderate"
        elif aqi <= 150:
            return "Unhealthy for Sensitive Groups"
        elif aqi <= 200:
            return "Unhealthy"
        elif aqi <= 300:
            return "Very Unhealthy"
        else:
            return "Hazardous"

    def _calculate_confidence(self, prediction: float, features: Dict) -> float:
        """Calculate prediction confidence based on data quality."""
        try:
            # Base confidence on model's feature importance
            importances = dict(zip(self.features, 
                                 self.model.feature_importances_))
            
            # Calculate data quality score
            data_quality = sum(1 for f in features if features[f] is not None) / len(features)
            
            # Weight the most important features more heavily
            key_features = ['pm25', 'pm10', 'emission_levels', 'co', 'no2']
            key_features_present = sum(1 for f in key_features if features.get(f) is not None)
            key_feature_quality = key_features_present / len(key_features)
            
            # Combined confidence score
            confidence = (0.7 * key_feature_quality + 0.3 * data_quality)
            
            return round(confidence, 2)
        except Exception as e:
            self.logger.warning(f"Error calculating confidence: {str(e)}")
            return 0.5

    def _analyze_contributing_factors(self, features: Dict) -> Dict[str, float]:
        """Analyze which factors contribute most to the prediction."""
        try:
            # Get feature importance from model
            importances = dict(zip(self.features, 
                                 self.model.feature_importances_))
            
            # Group factors by category
            factors = {
                "air_pollutants": sum(importances.get(f, 0) for f in 
                                    ['pm25', 'pm10', 'no2', 'co']),
                "weather_conditions": sum(importances.get(f, 0) for f in 
                                       ['temperature', 'humidity', 'wind_speed']),
                "industrial_impact": sum(importances.get(f, 0) for f in 
                                      ['industrial_activity_index', 'emission_levels']),
                "urban_factors": sum(importances.get(f, 0) for f in 
                                  ['daily_vehicle_count', 'urban_density'])
            }
            
            # Normalize to percentages
            total = sum(factors.values())
            return {k: round(v/total, 2) for k, v in factors.items()}
            
        except Exception as e:
            self.logger.error(f"Error analyzing factors: {str(e)}")
            return {}

    def predict(self, features: Dict[str, Any], hours: int = 1) -> Dict[str, Any]:
        """
        Make AQI prediction based on input features.
        
        Args:
            features: Dictionary of feature values
            hours: Number of hours to predict ahead
        
        Returns:
            Dictionary containing:
            - predicted_aqi: int
            - risk_level: str
            - confidence: float
            - contributing_factors: Dict[str, float]
        """
        try:
            # Prepare features
            df = pd.DataFrame([features])
            
            # Add missing columns
            for col in self.features:
                if col not in df.columns:
                    df[col] = 0
            
            # Ensure features are in the correct order
            ordered_df = pd.DataFrame(columns=self.features)
            for col in self.features:
                ordered_df[col] = df[col] if col in df else 0
            
            # Scale features
            scaled_features = self.scaler.transform(ordered_df)
            
            # Make prediction
            predicted_aqi = self.model.predict(scaled_features)[0]
            
            # Calculate confidence and analyze factors
            confidence = self._calculate_confidence(predicted_aqi, features)
            factors = self._analyze_contributing_factors(features)
            
            return {
                "aqi": round(predicted_aqi),
                "risk_level": self._get_risk_level(predicted_aqi),
                "prediction_for": f"next {hours} hour(s)",
                "confidence": confidence,
                "factors": factors,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Prediction error: {str(e)}")
            raise

    def batch_predict(self, features_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Make predictions for multiple feature sets."""
        return [self.predict(features) for features in features_list]