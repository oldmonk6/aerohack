#!/usr/bin/env python3
"""
Test script for the web API integration
"""

import requests
import json
import time

def test_web_api():
    """Test the Flask API endpoints"""
    base_url = "http://localhost:5000"
    
    print("Testing Rubik's Cube Web API...")
    print("=" * 50)
    
    # Test 1: Get initial state
    print("1. Testing /state endpoint...")
    try:
        response = requests.get(f"{base_url}/state")
        if response.status_code == 200:
            data = response.json()
            print("✅ State endpoint working")
            print(f"   Cube state: {list(data['state'].keys())}")
        else:
            print("❌ State endpoint failed")
    except Exception as e:
        print(f"❌ Error testing state endpoint: {e}")
    
    # Test 2: Scramble the cube
    print("\n2. Testing /scramble endpoint...")
    try:
        response = requests.post(f"{base_url}/scramble")
        if response.status_code == 200:
            data = response.json()
            print("✅ Scramble endpoint working")
            print(f"   Scramble moves: {' '.join(data['moves'])}")
        else:
            print("❌ Scramble endpoint failed")
    except Exception as e:
        print(f"❌ Error testing scramble endpoint: {e}")
    
    # Test 3: Solve with beginner's method
    print("\n3. Testing /beginner_solve endpoint...")
    try:
        response = requests.post(f"{base_url}/beginner_solve")
        if response.status_code == 200:
            data = response.json()
            print("✅ Beginner solve endpoint working")
            print(f"   Solution: {' '.join(data['solution'])}")
            print(f"   Method: {data.get('method', 'unknown')}")
        else:
            print("❌ Beginner solve endpoint failed")
    except Exception as e:
        print(f"❌ Error testing beginner solve endpoint: {e}")
    
    # Test 4: Reset the cube
    print("\n4. Testing /reset endpoint...")
    try:
        response = requests.post(f"{base_url}/reset")
        if response.status_code == 200:
            data = response.json()
            print("✅ Reset endpoint working")
        else:
            print("❌ Reset endpoint failed")
    except Exception as e:
        print(f"❌ Error testing reset endpoint: {e}")
    
    print("\n" + "=" * 50)
    print("Web API test complete!")

if __name__ == "__main__":
    # Wait a moment for the server to start
    print("Waiting for server to start...")
    time.sleep(3)
    test_web_api() 