#!/usr/bin/env python3
"""
Test script for FlavorSnap Analytics Dashboard
Tests all analytics API endpoints and validates responses
"""

import requests
import json
from datetime import datetime, timedelta

BASE_URL = "http://localhost:5000"

def test_endpoint(endpoint, description):
    """Test a single API endpoint"""
    print(f"\n🧪 Testing {description}")
    print(f"📍 Endpoint: {endpoint}")
    
    try:
        response = requests.get(f"{BASE_URL}{endpoint}")
        print(f"📊 Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Success: {description}")
            print(f"📋 Data keys: {list(data.keys()) if isinstance(data, dict) else type(data)}")
            
            # Validate specific endpoints
            if endpoint == "/analytics":
                validate_analytics_response(data)
            elif endpoint == "/analytics/usage":
                validate_usage_response(data)
            elif endpoint == "/analytics/performance":
                validate_performance_response(data)
            elif endpoint == "/analytics/engagement":
                validate_engagement_response(data)
                
        else:
            print(f"❌ Error: {response.status_code} - {response.text}")
            
    except requests.exceptions.ConnectionError:
        print(f"❌ Connection Error: Could not connect to {BASE_URL}")
        print("💡 Make sure the Flask server is running on port 5000")
    except Exception as e:
        print(f"❌ Unexpected Error: {str(e)}")

def validate_analytics_response(data):
    """Validate the main analytics response"""
    required_keys = ['usageStats', 'modelPerformance', 'userEngagement', 'statsCards', 'realTimeActivity']
    
    for key in required_keys:
        if key not in data:
            print(f"⚠️  Missing key: {key}")
        else:
            print(f"✅ Found key: {key} ({len(data[key]) if isinstance(data[key], list) else type(data[key])})")

def validate_usage_response(data):
    """Validate usage statistics response"""
    if isinstance(data, list) and len(data) > 0:
        sample = data[0]
        required_keys = ['date', 'requests', 'users', 'accuracy']
        
        for key in required_keys:
            if key in sample:
                print(f"✅ Usage data has {key}: {sample[key]}")
            else:
                print(f"⚠️  Missing usage key: {key}")

def validate_performance_response(data):
    """Validate model performance response"""
    if isinstance(data, list) and len(data) > 0:
        sample = data[0]
        required_keys = ['model', 'accuracy', 'inferenceTime', 'confidence']
        
        for key in required_keys:
            if key in sample:
                print(f"✅ Performance data has {key}: {sample[key]}")
            else:
                print(f"⚠️  Missing performance key: {key}")

def validate_engagement_response(data):
    """Validate user engagement response"""
    if isinstance(data, list) and len(data) > 0:
        sample = data[0]
        required_keys = ['category', 'value', 'color']
        
        for key in required_keys:
            if key in sample:
                print(f"✅ Engagement data has {key}: {sample[key]}")
            else:
                print(f"⚠️  Missing engagement key: {key}")

def test_date_filtering():
    """Test date range filtering"""
    print(f"\n🧪 Testing Date Range Filtering")
    
    # Generate test dates
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
    
    endpoint = f"/analytics/usage?start_date={start_date}&end_date={end_date}"
    print(f"📍 Endpoint: {endpoint}")
    
    try:
        response = requests.get(f"{BASE_URL}{endpoint}")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Date filtering works: {len(data)} records returned")
        else:
            print(f"❌ Date filtering failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Date filtering error: {str(e)}")

def test_export_functionality():
    """Test export functionality"""
    print(f"\n🧪 Testing Export Functionality")
    
    try:
        response = requests.get(f"{BASE_URL}/analytics/export")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Export works: {len(data)} fields exported")
            print(f"📋 Export keys: {list(data.keys())}")
        else:
            print(f"❌ Export failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Export error: {str(e)}")

def main():
    """Run all tests"""
    print("🚀 FlavorSnap Analytics Dashboard Test Suite")
    print("=" * 50)
    
    # Test all endpoints
    test_endpoint("/health", "Health Check")
    test_endpoint("/analytics", "Complete Analytics Data")
    test_endpoint("/analytics/usage", "Usage Statistics")
    test_endpoint("/analytics/performance", "Model Performance")
    test_endpoint("/analytics/engagement", "User Engagement")
    test_endpoint("/analytics/activity", "Real-time Activity")
    test_endpoint("/analytics/stats", "Stats Cards")
    
    # Test special features
    test_date_filtering()
    test_export_functionality()
    
    print("\n" + "=" * 50)
    print("🏁 Test Suite Complete")
    print("\n💡 Next Steps:")
    print("1. Start the frontend: cd frontend && npm run dev")
    print("2. Navigate to: http://localhost:3000/analytics")
    print("3. Verify all charts and components load correctly")

if __name__ == "__main__":
    main()
