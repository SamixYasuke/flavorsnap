import requests
import time
import os
from threading import Thread

BASE_URL = "http://127.0.0.1:5000"
API_KEY = os.environ.get('TEST_API_KEY', 'dev-api-key-12345678901234567890123456789012')

def test_health_check_exempt():
    print("\\n--- Testing Health Check (Exempt/1000 per hour) ---")
    for i in range(12):
        r = requests.get(f"{BASE_URL}/health")
        print(f"Health #{i+1}: {r.status_code}")
        # Health check should not be rate limited for small number of requests
        if r.status_code == 429:
            print("Failed: Health check was rate limited too early!")
            print("Headers:", r.headers)
            break
        
        # Check security headers
        if i == 0:  # Check headers on first request
            security_headers = ['X-Content-Type-Options', 'X-Frame-Options', 'X-XSS-Protection']
            missing_headers = [h for h in security_headers if h not in r.headers]
            if missing_headers:
                print(f"⚠️  Missing security headers: {missing_headers}")
            else:
                print("✅ Security headers present")

def test_predict_rate_limit():
    print("\\n--- Testing Predict Endpoint (100 per min limit) ---")
    # We will send 105 requests. Some should be 429
    success_count = 0
    ratelimited_count = 0
    auth_errors = 0
    
    with open(__file__, 'rb') as f:
        for i in range(105):
            try:
                # We do a mock file upload just to pass the first layer of validation
                files = {'image': ('test.jpg', f, 'image/jpeg')}
                headers = {'X-API-Key': API_KEY}
                r = requests.post(f"{BASE_URL}/predict", files=files, headers=headers)
                print(f"Predict #{i+1}: {r.status_code} - {r.json().get('error', 'success')}")
                
                # Check for rate limit headers
                if 'X-RateLimit-Limit' in r.headers:
                    limit_header = r.headers['X-RateLimit-Limit']
                    remaining_header = r.headers.get('X-RateLimit-Remaining', 'N/A')
                    print(f"   Limits: {limit_header}, Remaining: {remaining_header}")
                
                if r.status_code == 200:
                    success_count += 1
                elif r.status_code == 429:
                    ratelimited_count += 1
                elif r.status_code == 401:
                    auth_errors += 1
                    print(f"   Auth error: {r.json().get('error', 'Unknown auth error')}")
                    
            except Exception as e:
                print(f"Error: {e}")
                
    print(f"\\nResults: {success_count} successful, {ratelimited_count} rate limited, {auth_errors} auth errors")
    if ratelimited_count > 0:
        print("✅ Rate limiting logic is working as expected.")
    else:
        print("❌ Rate limiting failed. Expected some requests to be limited.")
    
    if auth_errors > 0:
        print("❌ Authentication errors detected. Check API key configuration.")
    else:
        print("✅ No authentication errors.")

def test_api_key_authentication():
    print("\\n--- Testing API Key Authentication ---")
    
    # Test without API key
    print("Testing without API key...")
    with open(__file__, 'rb') as f:
        files = {'image': ('test.jpg', f, 'image/jpeg')}
        r = requests.post(f"{BASE_URL}/predict", files=files)
        print(f"No API Key: {r.status_code} - {r.json().get('error', 'success')}")
        if r.status_code == 401:
            print("✅ API key authentication working")
        else:
            print("❌ API key authentication failed")
    
    # Test with invalid API key
    print("Testing with invalid API key...")
    with open(__file__, 'rb') as f:
        files = {'image': ('test.jpg', f, 'image/jpeg')}
        headers = {'X-API-Key': 'invalid-key-123'}
        r = requests.post(f"{BASE_URL}/predict", files=files, headers=headers)
        print(f"Invalid API Key: {r.status_code} - {r.json().get('error', 'success')}")
        if r.status_code == 401:
            print("✅ Invalid API key rejected")
        else:
            print("❌ Invalid API key accepted")

def test_input_validation():
    print("\\n--- Testing Input Validation ---")
    
    # Test no file
    print("Testing no file upload...")
    headers = {'X-API-Key': API_KEY}
    r = requests.post(f"{BASE_URL}/predict", headers=headers)
    print(f"No file: {r.status_code} - {r.json().get('error', 'success')}")
    if r.status_code == 400 and 'No image provided' in r.json().get('error', ''):
        print("✅ No file validation working")
    else:
        print("❌ No file validation failed")
    
    # Test invalid file type
    print("Testing invalid file type...")
    files = {'image': ('test.txt', b'fake content', 'text/plain')}
    headers = {'X-API-Key': API_KEY}
    r = requests.post(f"{BASE_URL}/predict", files=files, headers=headers)
    print(f"Invalid file type: {r.status_code} - {r.json().get('error', 'success')}")
    if r.status_code == 400 and ('Unsupported' in r.json().get('error', '') or 'file type' in r.json().get('error', '').lower()):
        print("✅ File type validation working")
    else:
        print("❌ File type validation failed")

def test_security_headers():
    print("\\n--- Testing Security Headers ---")
    r = requests.get(f"{BASE_URL}/health")
    
    security_headers = {
        'X-Content-Type-Options': 'nosniff',
        'X-Frame-Options': 'DENY',
        'X-XSS-Protection': '1; mode=block',
        'Content-Security-Policy': None,  # Just check presence
        'Referrer-Policy': None  # Just check presence
    }
    
    missing_headers = []
    for header, expected_value in security_headers.items():
        if header not in r.headers:
            missing_headers.append(header)
        elif expected_value and r.headers[header] != expected_value:
            print(f"⚠️  {header}: expected '{expected_value}', got '{r.headers[header]}'")
    
    if missing_headers:
        print(f"❌ Missing security headers: {missing_headers}")
    else:
        print("✅ All security headers present")

if __name__ == "__main__":
    # Wait for the server to spin up
    print("Waiting 2 seconds for server to start...")
    time.sleep(2)
    
    test_health_check_exempt()
    test_predict_rate_limit()
    test_api_key_authentication()
    test_input_validation()
    test_security_headers()
    
    print("\\n" + "="*50)
    print("Rate limiting and security tests completed!")
    print("Check the output above for any ❌ marks indicating issues.")
