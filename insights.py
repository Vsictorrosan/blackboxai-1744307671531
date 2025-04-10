"""
Enhanced AQI Insights Engine with 3-Level Recommendations
Project: Smart AQI Guardian
Author: Mithileshvinayak
Last Updated: 2025-04-10 13:15:30
"""

from typing import Dict, Any, List, Tuple
from datetime import datetime
import logging

class InsightsEngine:
    def __init__(self):
        self.logger = logging.getLogger('InsightsEngine')
        
    def _get_aqi_category(self, aqi: float) -> Tuple[str, str]:
        """Get AQI category and color code."""
        if aqi <= 50:
            return "Good", "green"
        elif aqi <= 100:
            return "Moderate", "yellow"
        elif aqi <= 150:
            return "Unhealthy for Sensitive Groups", "orange"
        elif aqi <= 200:
            return "Unhealthy", "red"
        elif aqi <= 300:
            return "Very Unhealthy", "purple"
        else:
            return "Hazardous", "maroon"

    def _get_individual_recommendations(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate recommendations for individuals."""
        recommendations = []
        aqi = data.get('aqi', 0)
        traffic = data.get('daily_vehicle_count', 0)
        spike_expected = data.get('predicted_spike', False)
        
        # Structure for each recommendation
        def add_rec(trigger: str, action: str, priority: str, icon: str):
            recommendations.append({
                "trigger": trigger,
                "action": action,
                "priority": priority,
                "icon": icon,
                "timestamp": datetime.utcnow().isoformat()
            })

        # AQI-based recommendations
        if aqi > 150:
            add_rec(
                "Unhealthy Air Quality",
                "Stay indoors, keep windows closed, use air purifiers if available",
                "HIGH",
                "🏠"
            )
        elif aqi > 100:
            add_rec(
                "Moderate Air Quality",
                "Wear masks outdoors, especially during peak hours",
                "MEDIUM",
                "😷"
            )

        # Traffic-based
        if traffic > 7000:
            add_rec(
                "Heavy Traffic Conditions",
                "Consider using public transport or alternative routes",
                "MEDIUM",
                "🚌"
            )

        # Spike predictions
        if spike_expected:
            add_rec(
                "AQI Spike Expected",
                "Plan outdoor activities for earlier/later time slots",
                "HIGH",
                "⚠️"
            )

        return recommendations

    def _get_government_recommendations(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate recommendations for government authorities."""
        recommendations = []
        aqi = data.get('aqi', 0)
        traffic = data.get('daily_vehicle_count', 0)
        industrial_activity = data.get('industrial_activity_index', 0)
        
        def add_rec(trigger: str, action: str, priority: str, department: str):
            recommendations.append({
                "trigger": trigger,
                "action": action,
                "priority": priority,
                "department": department,
                "timestamp": datetime.utcnow().isoformat()
            })

        # Traffic management
        if traffic > 7000 and aqi > 120:
            add_rec(
                "Traffic-induced AQI Spike",
                "Activate smart traffic management system, reroute heavy vehicles",
                "HIGH",
                "Traffic Control"
            )

        # Industrial zone management
        if industrial_activity > 75:
            add_rec(
                "High Industrial Activity",
                "Dispatch inspection team to industrial zones",
                "HIGH",
                "Environmental Protection"
            )

        # Continuous high AQI
        if aqi > 150:
            add_rec(
                "Sustained High AQI",
                "Consider implementing temporary vehicle restrictions",
                "CRITICAL",
                "City Administration"
            )

        return recommendations

    def _get_industrial_recommendations(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate recommendations for industrial zones."""
        recommendations = []
        aqi = data.get('aqi', 0)
        emission_levels = data.get('emission_levels', 0)
        
        def add_rec(trigger: str, action: str, priority: str, compliance_impact: str):
            recommendations.append({
                "trigger": trigger,
                "action": action,
                "priority": priority,
                "compliance_impact": compliance_impact,
                "timestamp": datetime.utcnow().isoformat()
            })

        # Emission-based recommendations
        if emission_levels > 80:
            add_rec(
                "Critical Emission Levels",
                "Immediately reduce production output or activate additional filters",
                "CRITICAL",
                "High"
            )
        elif emission_levels > 60:
            add_rec(
                "Elevated Emissions",
                "Review filter efficiency and maintenance schedule",
                "HIGH",
                "Medium"
            )

        # Zone-based AQI
        if aqi > 150:
            add_rec(
                "High AQI in Industrial Zone",
                "Implement emission control measures, check equipment",
                "HIGH",
                "Medium"
            )

        return recommendations

    def generate_recommendations(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive recommendations for all stakeholders."""
        try:
            aqi = data.get('aqi', 0)
            category, color = self._get_aqi_category(aqi)
            
            return {
                "status": {
                    "aqi": aqi,
                    "category": category,
                    "color_code": color,
                    "timestamp": datetime.utcnow().isoformat()
                },
                "recommendations": {
                    "individual": self._get_individual_recommendations(data),
                    "government": self._get_government_recommendations(data),
                    "industrial": self._get_industrial_recommendations(data)
                },
                "metadata": {
                    "location": data.get('location', 'Unknown'),
                    "data_confidence": data.get('confidence', 'Medium'),
                    "update_frequency": "5 minutes"
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error generating recommendations: {str(e)}")
            raise
