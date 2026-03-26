"""
Chart Utility Functions for FlavorSnap

This module provides utility functions for creating and managing confidence charts,
including data processing, color management, and responsive design helpers.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
import colorsys
import json

class ChartColorManager:
    """
    Manages color schemes and color coding for confidence charts.
    """
    
    def __init__(self):
        # Color palette for confidence levels
        self.confidence_colors = {
            'high': '#22c55e',      # Green
            'medium': '#eab308',    # Yellow  
            'low': '#ef4444',       # Red
            'default': '#e0e0e0'    # Gray
        }
        
        # Additional color schemes
        self.color_schemes = {
            'default': {
                'primary': '#3b82f6',
                'secondary': '#8b5cf6',
                'accent': '#ec4899',
                'success': '#22c55e',
                'warning': '#eab308',
                'error': '#ef4444'
            },
            'accessible': {
                'high': '#0066cc',
                'medium': '#ff9900',
                'low': '#cc0000',
                'default': '#666666'
            },
            'pastel': {
                'high': '#86efac',
                'medium': '#fde047',
                'low': '#fca5a5',
                'default': '#d1d5db'
            }
        }
    
    def get_confidence_color(self, confidence: float, scheme: str = 'default') -> str:
        """
        Get color based on confidence level.
        
        Args:
            confidence: Confidence score (0-1)
            scheme: Color scheme to use
            
        Returns:
            Color hex code
        """
        if confidence > 0.8:
            return self.confidence_colors['high']
        elif confidence >= 0.6:
            return self.confidence_colors['medium']
        else:
            return self.confidence_colors['low']
    
    def generate_gradient_colors(self, base_color: str, steps: int = 10) -> List[str]:
        """
        Generate a gradient of colors from a base color.
        
        Args:
            base_color: Base hex color
            steps: Number of gradient steps
            
        Returns:
            List of hex color codes
        """
        # Convert hex to RGB
        base_color = base_color.lstrip('#')
        r, g, b = tuple(int(base_color[i:i+2], 16) for i in (0, 2, 4))
        
        # Convert to HSV
        h, s, v = colorsys.rgb_to_hsv(r/255.0, g/255.0, b/255.0)
        
        # Generate gradient
        colors = []
        for i in range(steps):
            # Vary the value (brightness)
            new_v = v * (0.3 + 0.7 * (i / (steps - 1)))
            new_r, new_g, new_b = colorsys.hsv_to_rgb(h, s, new_v)
            colors.append(f"#{int(new_r*255):02x}{int(new_g*255):02x}{int(new_b*255):02x}")
        
        return colors
    
    def get_contrasting_text_color(self, bg_color: str) -> str:
        """
        Get a contrasting text color for a given background.
        
        Args:
            bg_color: Background hex color
            
        Returns:
            Contrasting text color (black or white)
        """
        # Convert hex to RGB
        bg_color = bg_color.lstrip('#')
        r, g, b = tuple(int(bg_color[i:i+2], 16) for i in (0, 2, 4))
        
        # Calculate luminance
        luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255
        
        return '#000000' if luminance > 0.5 else '#ffffff'


class ChartDataProcessor:
    """
    Processes and formats data for confidence charts.
    """
    
    def __init__(self):
        self.class_names = ['Akara', 'Bread', 'Egusi', 'Moi Moi', 'Rice and Stew', 'Yam']
    
    def format_probabilities(self, probabilities: Dict[str, float]) -> Dict[str, Any]:
        """
        Format raw probabilities for chart display.
        
        Args:
            probabilities: Raw probability dictionary
            
        Returns:
            Formatted data dictionary
        """
        formatted_data = {
            'classes': [],
            'probabilities': [],
            'percentages': [],
            'colors': [],
            'metadata': {}
        }
        
        # Sort by probability (descending)
        sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
        
        color_manager = ChartColorManager()
        
        for class_name, prob in sorted_probs:
            formatted_data['classes'].append(class_name)
            formatted_data['probabilities'].append(prob)
            formatted_data['percentages'].append(prob * 100)
            formatted_data['colors'].append(color_manager.get_confidence_color(prob))
        
        # Add metadata
        formatted_data['metadata'] = {
            'top_class': sorted_probs[0][0] if sorted_probs else None,
            'top_confidence': sorted_probs[0][1] if sorted_probs else 0,
            'average_confidence': np.mean(list(probabilities.values())),
            'entropy': self._calculate_entropy(probabilities)
        }
        
        return formatted_data
    
    def _calculate_entropy(self, probabilities: Dict[str, float]) -> float:
        """
        Calculate the entropy of the probability distribution.
        
        Args:
            probabilities: Probability dictionary
            
        Returns:
            Entropy value
        """
        probs = np.array(list(probabilities.values()))
        # Avoid log(0)
        probs = probs[probs > 0]
        return -np.sum(probs * np.log2(probs))
    
    def create_chart_dataframe(self, probabilities: Dict[str, float]) -> pd.DataFrame:
        """
        Create a pandas DataFrame for chart visualization.
        
        Args:
            probabilities: Probability dictionary
            
        Returns:
            Formatted DataFrame
        """
        data = []
        color_manager = ChartColorManager()
        
        for class_name in self.class_names:
            if class_name in probabilities:
                prob = probabilities[class_name]
                data.append({
                    'Class': class_name,
                    'Probability': prob,
                    'Percentage': prob * 100,
                    'Color': color_manager.get_confidence_color(prob),
                    'Confidence_Level': self._get_confidence_level(prob)
                })
            else:
                data.append({
                    'Class': class_name,
                    'Probability': 0.0,
                    'Percentage': 0.0,
                    'Color': color_manager.confidence_colors['default'],
                    'Confidence_Level': 'None'
                })
        
        # Sort by probability (descending)
        df = pd.DataFrame(data)
        df = df.sort_values('Probability', ascending=False).reset_index(drop=True)
        
        return df
    
    def _get_confidence_level(self, confidence: float) -> str:
        """Get confidence level category."""
        if confidence > 0.8:
            return 'High'
        elif confidence >= 0.6:
            return 'Medium'
        else:
            return 'Low'


class ResponsiveChartHelper:
    """
    Helper functions for responsive chart design.
    """
    
    def __init__(self):
        self.breakpoints = {
            'mobile': 480,
            'tablet': 768,
            'desktop': 1024,
            'large': 1200
        }
    
    def get_responsive_dimensions(self, container_width: int) -> Dict[str, int]:
        """
        Get responsive chart dimensions based on container width.
        
        Args:
            container_width: Width of the container in pixels
            
        Returns:
            Dictionary with responsive dimensions
        """
        if container_width < self.breakpoints['mobile']:
            return {
                'chart_height': 300,
                'bar_height': 35,
                'font_size': 12,
                'margin': 8
            }
        elif container_width < self.breakpoints['tablet']:
            return {
                'chart_height': 350,
                'bar_height': 40,
                'font_size': 13,
                'margin': 10
            }
        elif container_width < self.breakpoints['desktop']:
            return {
                'chart_height': 400,
                'bar_height': 45,
                'font_size': 14,
                'margin': 12
            }
        else:
            return {
                'chart_height': 450,
                'bar_height': 50,
                'font_size': 15,
                'margin': 16
            }
    
    def generate_responsive_css(self) -> str:
        """
        Generate responsive CSS for charts.
        
        Returns:
            CSS string for responsive design
        """
        return f"""
        /* Responsive Chart Styles */
        @media (max-width: {self.breakpoints['mobile']}px) {{
            .confidence-chart-container {{
                padding: 8px;
            }}
            
            .chart-bar-container {{
                margin-bottom: 6px;
            }}
            
            .chart-label {{
                font-size: 12px;
            }}
            
            .chart-bar {{
                height: 25px;
            }}
            
            .legend {{
                font-size: 10px;
                flex-direction: column;
                gap: 4px;
            }}
        }}
        
        @media (min-width: {self.breakpoints['mobile']}px) and (max-width: {self.breakpoints['tablet']}px) {{
            .confidence-chart-container {{
                padding: 12px;
            }}
            
            .chart-bar {{
                height: 30px;
            }}
            
            .chart-label {{
                font-size: 13px;
            }}
        }}
        
        @media (min-width: {self.breakpoints['tablet']}px) {{
            .confidence-chart-container {{
                padding: 16px;
            }}
            
            .chart-bar {{
                height: 35px;
            }}
            
            .chart-label {{
                font-size: 14px;
            }}
        }}
        
        @media (min-width: {self.breakpoints['desktop']}px) {{
            .confidence-chart-container {{
                max-width: 800px;
                margin: 0 auto;
            }}
        }}
        """


class ChartAnimationHelper:
    """
    Helper functions for chart animations and transitions.
    """
    
    def __init__(self):
        self.default_duration = 800  # milliseconds
        self.easing_functions = {
            'ease-in-out': 'cubic-bezier(0.4, 0, 0.2, 1)',
            'ease-out': 'cubic-bezier(0, 0, 0.2, 1)',
            'ease-in': 'cubic-bezier(0.4, 0, 1, 1)',
            'bounce': 'cubic-bezier(0.68, -0.55, 0.265, 1.55)'
        }
    
    def generate_animation_css(self) -> str:
        """
        Generate CSS for chart animations.
        
        Returns:
            CSS animation string
        """
        return f"""
        /* Chart Animations */
        .chart-bar {{
            transition: width {self.default_duration}ms ease-in-out,
                        background-color {self.default_duration}ms ease-in-out,
                        transform {self.default_duration}ms ease-in-out;
        }}
        
        .chart-bar:hover {{
            transform: scaleX(1.02);
            filter: brightness(1.1);
        }}
        
        .fade-in-up {{
            opacity: 0;
            animation: fadeInUp 0.6s ease-out forwards;
        }}
        
        @keyframes fadeInUp {{
            from {{
                opacity: 0;
                transform: translateY(30px);
            }}
            to {{
                opacity: 1;
                transform: translateY(0);
            }}
        }}
        
        .slide-in-left {{
            opacity: 0;
            animation: slideInLeft 0.5s ease-out forwards;
        }}
        
        @keyframes slideInLeft {{
            from {{
                opacity: 0;
                transform: translateX(-50px);
            }}
            to {{
                opacity: 1;
                transform: translateX(0);
            }}
        }}
        
        .pulse {{
            animation: pulse 2s infinite;
        }}
        
        @keyframes pulse {{
            0%, 100% {{
                opacity: 1;
            }}
            50% {{
                opacity: 0.7;
            }}
        }}
        
        /* Staggered animations for chart bars */
        .chart-bar-container:nth-child(1) {{ animation-delay: 0.1s; }}
        .chart-bar-container:nth-child(2) {{ animation-delay: 0.2s; }}
        .chart-bar-container:nth-child(3) {{ animation-delay: 0.3s; }}
        .chart-bar-container:nth-child(4) {{ animation-delay: 0.4s; }}
        .chart-bar-container:nth-child(5) {{ animation-delay: 0.5s; }}
        .chart-bar-container:nth-child(6) {{ animation-delay: 0.6s; }}
        """
    
    def generate_transition_javascript(self) -> str:
        """
        Generate JavaScript for smooth chart transitions.
        
        Returns:
            JavaScript code string
        """
        return """
        // Chart transition utilities
        class ChartTransitionManager {
            constructor() {
                this.activeTransitions = new Map();
            }
            
            // Animate bar width changes
            animateBarWidth(element, targetWidth, duration = 800) {
                if (this.activeTransitions.has(element)) {
                    cancelAnimationFrame(this.activeTransitions.get(element));
                }
                
                const startWidth = parseFloat(element.style.width) || 0;
                const startTime = performance.now();
                
                const animate = (currentTime) => {
                    const elapsed = currentTime - startTime;
                    const progress = Math.min(elapsed / duration, 1);
                    
                    // Easing function (ease-in-out)
                    const easeProgress = progress < 0.5 
                        ? 2 * progress * progress 
                        : 1 - Math.pow(-2 * progress + 2, 2) / 2;
                    
                    const currentWidth = startWidth + (targetWidth - startWidth) * easeProgress;
                    element.style.width = currentWidth + '%';
                    
                    if (progress < 1) {
                        const animationId = requestAnimationFrame(animate);
                        this.activeTransitions.set(element, animationId);
                    } else {
                        this.activeTransitions.delete(element);
                    }
                };
                
                requestAnimationFrame(animate);
            }
            
            // Animate color changes
            animateColorChange(element, targetColor, duration = 800) {
                element.style.transition = `background-color ${duration}ms ease-in-out`;
                element.style.backgroundColor = targetColor;
                
                setTimeout(() => {
                    element.style.transition = '';
                }, duration);
            }
            
            // Add hover effects
            addHoverEffects(chartContainer) {
                const bars = chartContainer.querySelectorAll('.chart-bar');
                
                bars.forEach(bar => {
                    bar.addEventListener('mouseenter', () => {
                        bar.style.transform = 'scaleX(1.02)';
                        bar.style.filter = 'brightness(1.1)';
                    });
                    
                    bar.addEventListener('mouseleave', () => {
                        bar.style.transform = 'scaleX(1)';
                        bar.style.filter = 'brightness(1)';
                    });
                });
            }
        }
        
        // Global transition manager instance
        window.chartTransitionManager = new ChartTransitionManager();
        """


# Utility functions for common chart operations
def create_chart_tooltip_data(probabilities: Dict[str, float]) -> Dict[str, str]:
    """
    Create tooltip data for chart interactions.
    
    Args:
        probabilities: Probability dictionary
        
    Returns:
        Tooltip data dictionary
    """
    tooltip_data = {}
    for class_name, prob in probabilities.items():
        percentage = prob * 100
        confidence_level = "High" if prob > 0.8 else "Medium" if prob >= 0.6 else "Low"
        tooltip_data[class_name] = f"{class_name}: {percentage:.1f}% ({confidence_level} confidence)"
    
    return tooltip_data


def validate_chart_data(probabilities: Dict[str, float]) -> Tuple[bool, str]:
    """
    Validate chart probability data.
    
    Args:
        probabilities: Probability dictionary to validate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not probabilities:
        return False, "No probability data provided"
    
    if not isinstance(probabilities, dict):
        return False, "Probabilities must be a dictionary"
    
    # Check if probabilities sum to approximately 1
    total_prob = sum(probabilities.values())
    if abs(total_prob - 1.0) > 0.1:  # Allow some tolerance
        return False, f"Probabilities sum to {total_prob:.3f}, expected ~1.0"
    
    # Check individual probability values
    for class_name, prob in probabilities.items():
        if not isinstance(prob, (int, float)):
            return False, f"Probability for {class_name} must be a number"
        if prob < 0 or prob > 1:
            return False, f"Probability for {class_name} ({prob}) must be between 0 and 1"
    
    return True, ""


def export_chart_data(probabilities: Dict[str, float], predicted_class: str, format: str = 'json') -> str:
    """
    Export chart data in various formats.
    
    Args:
        probabilities: Probability dictionary
        predicted_class: Top predicted class
        format: Export format ('json', 'csv', 'xml')
        
    Returns:
        Exported data string
    """
    export_data = {
        'predicted_class': predicted_class,
        'probabilities': probabilities,
        'metadata': {
            'total_classes': len(probabilities),
            'top_confidence': max(probabilities.values()),
            'average_confidence': sum(probabilities.values()) / len(probabilities),
            'export_timestamp': pd.Timestamp.now().isoformat()
        }
    }
    
    if format.lower() == 'json':
        return json.dumps(export_data, indent=2)
    elif format.lower() == 'csv':
        df = pd.DataFrame(list(probabilities.items()), columns=['Class', 'Probability'])
        return df.to_csv(index=False)
    elif format.lower() == 'xml':
        # Simple XML export
        xml_lines = ['<?xml version="1.0" encoding="UTF-8"?>', '<chart_data>']
        xml_lines.append(f'  <predicted_class>{predicted_class}</predicted_class>')
        xml_lines.append('  <probabilities>')
        for class_name, prob in probabilities.items():
            xml_lines.append(f'    <class name="{class_name}" probability="{prob:.4f}"/>')
        xml_lines.append('  </probabilities>')
        xml_lines.append('</chart_data>')
        return '\n'.join(xml_lines)
    else:
        raise ValueError(f"Unsupported export format: {format}")


# Global instances for easy access
color_manager = ChartColorManager()
data_processor = ChartDataProcessor()
responsive_helper = ResponsiveChartHelper()
animation_helper = ChartAnimationHelper()
