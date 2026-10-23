"""
CLA 4 - Task 5: API Test Client (client.py)
Description:
    Automated client script to test the FastAPI model serving endpoints:
    1. GET /health
    2. POST /predict (Valid Request 1 - High Survival Likelihood Passenger)
    3. POST /predict (Valid Request 2 - Low Survival Likelihood Passenger)
    4. POST /predict (Invalid Request - Out-of-range values to verify Pydantic 422 error)
"""

import json
import requests

BASE_URL = "http://127.0.0.1:8000"


def test_health():
    print("=" * 60)
    print("TEST 1: GET /health")
    print("=" * 60)
    url = f"{BASE_URL}/health"
    print(f"Request: GET {url}")
    try:
        response = requests.get(url, timeout=5)
        print(f"Status Code: {response.status_code}")
        print("Response JSON:")
        print(json.dumps(response.json(), indent=4))
        assert response.status_code == 200, "Health check failed!"
        print("--> Health Check: PASSED\n")
    except requests.exceptions.ConnectionError:
        print("[ERROR] Could not connect to FastAPI server at http://127.0.0.1:8000")
        print("Please start the server first using: uvicorn app:app --port 8000\n")
        raise


def test_valid_prediction_1():
    print("=" * 60)
    print("TEST 2: POST /predict (Valid Case 1: 1st Class Female)")
    print("=" * 60)
    url = f"{BASE_URL}/predict"
    payload = {
        "Pclass": 1,
        "Sex": "female",
        "Age": 28.0,
        "SibSp": 0,
        "Parch": 0,
        "Fare": 85.50,
        "Embarked": "C"
    }
    print(f"Request: POST {url}")
    print("Payload:")
    print(json.dumps(payload, indent=4))

    response = requests.post(url, json=payload, timeout=5)
    print(f"\nStatus Code: {response.status_code}")
    print("Response JSON:")
    print(json.dumps(response.json(), indent=4))
    assert response.status_code == 200, "Prediction failed!"
    print("--> Valid Prediction 1: PASSED\n")


def test_valid_prediction_2():
    print("=" * 60)
    print("TEST 3: POST /predict (Valid Case 2: 3rd Class Male)")
    print("=" * 60)
    url = f"{BASE_URL}/predict"
    payload = {
        "Pclass": 3,
        "Sex": "male",
        "Age": 24.0,
        "SibSp": 2,
        "Parch": 0,
        "Fare": 24.15,
        "Embarked": "S"
    }
    print(f"Request: POST {url}")
    print("Payload:")
    print(json.dumps(payload, indent=4))

    response = requests.post(url, json=payload, timeout=5)
    print(f"\nStatus Code: {response.status_code}")
    print("Response JSON:")
    print(json.dumps(response.json(), indent=4))
    assert response.status_code == 200, "Prediction failed!"
    print("--> Valid Prediction 2: PASSED\n")


def test_invalid_prediction():
    print("=" * 60)
    print("TEST 4: POST /predict (Invalid Case: Pydantic Validation Error)")
    print("=" * 60)
    url = f"{BASE_URL}/predict"
    # Invalid Pclass (5 instead of 1-3), invalid Sex ('unknown'), negative fare (-10)
    invalid_payload = {
        "Pclass": 5,
        "Sex": "unknown",
        "Age": -5.0,
        "SibSp": -1,
        "Parch": 0,
        "Fare": -10.0,
        "Embarked": "Z"
    }
    print(f"Request: POST {url}")
    print("Payload (Intentional invalid schema to trigger 422):")
    print(json.dumps(invalid_payload, indent=4))

    response = requests.post(url, json=invalid_payload, timeout=5)
    print(f"\nStatus Code: {response.status_code} (Expected: 422 Unprocessable Entity)")
    print("Response JSON:")
    print(json.dumps(response.json(), indent=4))
    assert response.status_code == 422, "Validation did not catch errors!"
    print("--> Pydantic Validation Handling: PASSED\n")


def main():
    print("============================================================")
    print("           FASTAPI TITANIC MODEL SERVICE CLIENT SUITE")
    print("============================================================\n")
    test_health()
    test_valid_prediction_1()
    test_valid_prediction_2()
    test_invalid_prediction()
    print("============================================================")
    print("          ALL FASTAPI CLIENT TESTS PASSED SUCCESSFULLY!     ")
    print("============================================================")


if __name__ == "__main__":
    main()
