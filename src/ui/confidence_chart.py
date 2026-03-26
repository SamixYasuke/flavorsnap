"""
Confidence Chart Visualization Component for FlavorSnap

This module provides visual confidence score displays for food classification predictions.
It creates interactive bar charts showing probabilities for all food classes.
"""

import panel as pn
import param
import pandas as pd
import numpy as np
from typing import Dict, Any, List
import json

# Initialize Panel extensions for charts
pn.extension('tabulator')

class ConfidenceChart(param.Parameterized):
    """
    Interactive confidence chart component for displaying classification probabilities.
    
    Features:
    - Bar chart visualization for all 6 food classes
    - Color coding based on confidence levels
    - Animated transitions
    - Hover tooltips with exact percentages
    - Responsive design for mobile devices
    """
    
    # Parameters
    probabilities = param.Dict(default={}, doc="Dictionary of class probabilities")
    predicted_class = param.String(default="", doc="Top predicted class")
    chart_height = param.Integer(default=400, doc="Height of the chart in pixels")
    
    def __init__(self, **params):
        super().__init__(**params)
        self.class_names = ['Akara', 'Bread', 'Egusi', 'Moi Moi', 'Rice and Stew', 'Yam']
        self._setup_chart()
    
    def _setup_chart(self):
        """Initialize the chart components."""
        # Create DataFrame for chart data
        self.chart_data = pd.DataFrame({
            'Class': self.class_names,
            'Probability': [0.0] * len(self.class_names),
            'Color': ['#e0e0e0'] * len(self.class_names)
        })
        
        # Create the bar chart using Panel's Chart components
        self.chart_pane = pn.pane.DataFrame(
            self.chart_data,
            show_index=False,
            sizing_mode='stretch_width',
            height=self.chart_height
        )
        
        # Create custom HTML chart for better control
        self.html_chart = pn.pane.HTML(
            self._generate_chart_html(),
            sizing_mode='stretch_width',
            height=self.chart_height
        )
        
        # Status message
        self.status_pane = pn.pane.Markdown(
            "📊 Upload and classify an image to see confidence scores",
            sizing_mode='stretch_width'
        )
    
    def _get_color_for_confidence(self, confidence: float) -> str:
        """
        Get color based on confidence level.
        
        Args:
            confidence: Confidence score (0-1)
            
        Returns:
            Color code string
        """
        if confidence > 0.8:
            return '#22c55e'  # Green
        elif confidence >= 0.6:
            return '#eab308'  # Yellow  
        else:
            return '#ef4444'  # Red
    
    def _generate_chart_html(self) -> str:
        """
        Generate HTML for the confidence chart with animations and tooltips.
        
        Returns:
            HTML string for the chart
        """
        if not self.probabilities:
            return """
            <div class="confidence-chart-container">
                <div class="no-data-message">
                    <p>📊 Upload and classify an image to see confidence scores</p>
                </div>
            </div>
            """
        
        # Sort classes by probability (descending)
        sorted_classes = sorted(
            self.probabilities.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        max_prob = max(self.probabilities.values()) if self.probabilities else 1.0
        
        chart_bars = []
        for class_name, probability in sorted_classes:
            color = self._get_color_for_confidence(probability)
            percentage = probability * 100
            bar_width = (probability / max_prob) * 100 if max_prob > 0 else 0
            
            # Add highlight for top prediction
            is_top = class_name == self.predicted_class
            highlight_class = "top-prediction" if is_top else ""
            
            chart_bars.append(f"""
            <div class="chart-bar-container {highlight_class}">
                <div class="chart-label">
                    <span class="class-name">{class_name}</span>
                    <span class="percentage">{percentage:.1f}%</span>
                </div>
                <div class="chart-bar-wrapper">
                    <div class="chart-bar" 
                         style="width: {bar_width}%; background-color: {color};"
                         data-class="{class_name}"
                         data-confidence="{percentage:.2f}%">
                    </div>
                </div>
            </div>
            """)
        
        return f"""
        <div class="confidence-chart-container">
            <div class="chart-header">
                <h3>🎯 Confidence Scores</h3>
                <div class="legend">
                    <span class="legend-item high">>80% High</span>
                    <span class="legend-item medium">60-80% Medium</span>
                    <span class="legend-item low"><60% Low</span>
                </div>
            </div>
            <div class="chart-bars">
                {''.join(chart_bars)}
            </div>
            <div class="chart-footer">
                <p class="top-prediction-text">
                    🏆 Top Prediction: <strong>{self.predicted_class}</strong>
                </p>
            </div>
        </div>
        """
    
    def update_predictions(self, probabilities: Dict[str, float], predicted_class: str):
        """
        Update the chart with new prediction data.
        
        Args:
            probabilities: Dictionary of class probabilities
            predicted_class: Top predicted class name
        """
        self.probabilities = probabilities
        self.predicted_class = predicted_class
        
        # Update chart data
        chart_data = []
        for class_name in self.class_names:
            if class_name in probabilities:
                prob = probabilities[class_name]
                color = self._get_color_for_confidence(prob)
                chart_data.append({
                    'Class': class_name,
                    'Probability': prob,
                    'Percentage': prob * 100,
                    'Color': color
                })
        
        # Sort by probability (descending)
        chart_data.sort(key=lambda x: x['Probability'], reverse=True)
        self.chart_data = pd.DataFrame(chart_data)
        
        # Update HTML chart
        self.html_chart.object = self._generate_chart_html()
        
        # Update status
        confidence = probabilities.get(predicted_class, 0) * 100
        self.status_pane.object = f"""
        ✅ **Classification Complete**
        
        🏆 **Top Prediction:** {predicted_class}
        📊 **Confidence:** {confidence:.1f}%
        
        Hover over the chart bars to see exact percentages.
        """
    
    def create_layout(self) -> pn.Column:
        """
        Create the complete layout for the confidence chart component.
        
        Returns:
            Panel Column with the chart layout
        """
        return pn.Column(
            self.status_pane,
            pn.layout.Divider(),
            self.html_chart,
            sizing_mode='stretch_width',
            css_classes=['confidence-chart-wrapper']
        )
    
    def reset(self):
        """Reset the chart to initial state."""
        self.probabilities = {}
        self.predicted_class = ""
        self.html_chart.object = self._generate_chart_html()
        self.status_pane.object = "📊 Upload and classify an image to see confidence scores"


class AnimatedConfidenceChart(ConfidenceChart):
    """
    Extended confidence chart with smooth animations and transitions.
    """
    
    def __init__(self, **params):
        super().__init__(**params)
        self.animation_duration = 800  # milliseconds
        self.previous_probabilities = {}
    
    def _generate_animated_chart_html(self) -> str:
        """Generate HTML with animation support."""
        base_html = self._generate_chart_html()
        
        # Add animation styles and scripts
        animation_script = f"""
        <style>
        .chart-bar {{
            transition: width {self.animation_duration}ms ease-in-out,
                        background-color {self.animation_duration}ms ease-in-out;
        }}
        
        .chart-bar-container {{
            opacity: 0;
            animation: fadeInUp 0.5s ease-out forwards;
        }}
        
        .chart-bar-container:nth-child(1) {{ animation-delay: 0.1s; }}
        .chart-bar-container:nth-child(2) {{ animation-delay: 0.2s; }}
        .chart-bar-container:nth-child(3) {{ animation-delay: 0.3s; }}
        .chart-bar-container:nth-child(4) {{ animation-delay: 0.4s; }}
        .chart-bar-container:nth-child(5) {{ animation-delay: 0.5s; }}
        .chart-bar-container:nth-child(6) {{ animation-delay: 0.6s; }}
        
        @keyframes fadeInUp {{
            from {{
                opacity: 0;
                transform: translateY(20px);
            }}
            to {{
                opacity: 1;
                transform: translateY(0);
            }}
        }}
        
        .top-prediction {{
            box-shadow: 0 4px 12px rgba(34, 197, 94, 0.3);
            border-left: 4px solid #22c55e;
            padding-left: 12px;
            margin-left: -16px;
        }}
        </style>
        """
        
        return base_html + animation_script
    
    def update_predictions(self, probabilities: Dict[str, float], predicted_class: str):
        """Update with animation support."""
        self.previous_probabilities = self.probabilities.copy()
        super().update_predictions(probabilities, predicted_class)
        self.html_chart.object = self._generate_animated_chart_html()


# Convenience function for creating chart instances
def create_confidence_chart(animate: bool = True) -> ConfidenceChart:
    """
    Factory function to create a confidence chart instance.
    
    Args:
        animate: Whether to use the animated version
        
    Returns:
        ConfidenceChart instance
    """
    if animate:
        return AnimatedConfidenceChart()
    else:
        return ConfidenceChart()


# Panel extension registration
def register_confidence_chart_extensions():
    """Register the confidence chart components with Panel."""
    if 'confidence_chart' not in pn.config.js_files:
        pn.config.js_files['confidence_chart'] = [
            'static/js/charts.js'
        ]
    
    if 'confidence_chart' not in pn.config.css_files:
        pn.config.css_files['confidence_chart'] = [
            'static/css/charts.css'
        ]
