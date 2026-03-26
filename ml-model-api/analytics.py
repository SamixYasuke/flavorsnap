from flask import Flask, request, jsonify
from datetime import datetime, timedelta
import random
import json

class AnalyticsAPI:
    def __init__(self):
        self.usage_data = []
        self.model_performance = []
        self.user_engagement = []
        self.generate_mock_data()
    
    def generate_mock_data(self):
        # Generate usage statistics for the last 30 days
        for i in range(30):
            date = datetime.now() - timedelta(days=i)
            self.usage_data.append({
                'date': date.strftime('%Y-%m-%d'),
                'requests': random.randint(100, 600),
                'users': random.randint(50, 250),
                'accuracy': round(random.uniform(85, 95), 1)
            })
        
        # Model performance data
        self.model_performance = [
            {
                'model': 'ResNet18',
                'accuracy': 94.2,
                'inferenceTime': 234,
                'confidence': 87.5
            },
            {
                'model': 'ResNet34',
                'accuracy': 95.1,
                'inferenceTime': 312,
                'confidence': 89.2
            },
            {
                'model': 'EfficientNet',
                'accuracy': 93.8,
                'inferenceTime': 189,
                'confidence': 86.1
            }
        ]
        
        # User engagement data
        self.user_engagement = [
            {'category': 'Akara', 'value': 23, 'color': '#FF6B6B'},
            {'category': 'Bread', 'value': 19, 'color': '#4ECDC4'},
            {'category': 'Egusi', 'value': 17, 'color': '#45B7D1'},
            {'category': 'Moi Moi', 'value': 21, 'color': '#96CEB4'},
            {'category': 'Rice and Stew', 'value': 12, 'color': '#FFEAA7'},
            {'category': 'Yam', 'value': 8, 'color': '#DDA0DD'}
        ]
    
    def get_usage_stats(self, start_date=None, end_date=None):
        if start_date and end_date:
            filtered_data = [
                item for item in self.usage_data
                if start_date <= item['date'] <= end_date
            ]
            return filtered_data
        return self.usage_data
    
    def get_model_performance(self):
        return self.model_performance
    
    def get_user_engagement(self):
        return self.user_engagement
    
    def get_real_time_activity(self):
        activities = [
            {
                'id': '1',
                'type': 'classification',
                'title': 'Classification Request',
                'description': 'Jollof Rice - 95.2% confidence',
                'timestamp': '2 min ago'
            },
            {
                'id': '2',
                'type': 'model_update',
                'title': 'Model Update',
                'description': 'Accuracy improved to 94.5%',
                'timestamp': '15 min ago'
            },
            {
                'id': '3',
                'type': 'alert',
                'title': 'High Traffic Alert',
                'description': '150+ requests in last hour',
                'timestamp': '1 hour ago'
            }
        ]
        return activities
    
    def get_stats_cards(self):
        return [
            {
                'title': 'Total Requests',
                'value': '12,847',
                'change': '+12.5%',
                'icon': 'activity',
                'color': 'bg-blue-500'
            },
            {
                'title': 'Active Users',
                'value': '3,421',
                'change': '+8.2%',
                'icon': 'users',
                'color': 'bg-green-500'
            },
            {
                'title': 'Avg Accuracy',
                'value': '94.2%',
                'change': '+2.1%',
                'icon': 'check-circle',
                'color': 'bg-purple-500'
            },
            {
                'title': 'Response Time',
                'value': '234ms',
                'change': '-15ms',
                'icon': 'clock',
                'color': 'bg-orange-500'
            }
        ]
    
    def export_data(self, start_date=None, end_date=None):
        export_data = {
            'usageData': self.get_usage_stats(start_date, end_date),
            'modelPerformance': self.get_model_performance(),
            'userEngagement': self.get_user_engagement(),
            'statsCards': self.get_stats_cards(),
            'realTimeActivity': self.get_real_time_activity(),
            'exportDate': datetime.now().isoformat(),
            'dateRange': {
                'start': start_date,
                'end': end_date
            }
        }
        return export_data

analytics = AnalyticsAPI()
