import requests
import json

BASE_URL = "http://127.0.0.1:8001"

print("--- 1. CORS Check ---")
try:
    headers = {
        "Origin": "http://localhost:5173",
        "Access-Control-Request-Method": "POST",
        "Access-Control-Request-Headers": "Content-Type"
    }
    # Preflight request
    resp = requests.options(f"{BASE_URL}/predict", headers=headers)
    print(f"Preflight status code: {resp.status_code}")
    print(f"Access-Control-Allow-Origin: {resp.headers.get('Access-Control-Allow-Origin')}")
    
    # Actual request
    resp2 = requests.post(
        f"{BASE_URL}/predict",
        json={"symptoms": ["itching", "skin_rash"]},
        headers={"Origin": "http://localhost:5173"}
    )
    print(f"POST status code: {resp2.status_code}")
    print(f"Access-Control-Allow-Origin: {resp2.headers.get('Access-Control-Allow-Origin')}")
except Exception as e:
    print(f"CORS Check Failed: {e}")

print("\n--- 2. Confidence Score Variation ---")
test_cases = [
    ["itching", "skin_rash", "nodal_skin_eruptions"],
    ["vomiting", "headache", "nausea", "spinning_movements"],
    ["joint_pain", "muscle_weakness", "stiff_neck", "swelling_joints"]
]

for i, symptoms in enumerate(test_cases):
    print(f"\nTest Case {i+1} Symptoms: {symptoms}")
    resp = requests.post(f"{BASE_URL}/predict", json={"symptoms": symptoms})
    if resp.status_code == 200:
        results = resp.json()
        for j, res in enumerate(results):
            print(f"  #{j+1} {res['disease']}: {res['confidence']:.2f}%")
    else:
        print(f"  Failed with status {resp.status_code}")

print("\n--- 3. Exact Response Schema ---")
print("POST /predict (example):")
resp_pred = requests.post(f"{BASE_URL}/predict", json={"symptoms": ["itching"]})
print(json.dumps(resp_pred.json(), indent=2))

print("\nGET /symptoms (first 5 elements):")
resp_sym = requests.get(f"{BASE_URL}/symptoms")
print(json.dumps(resp_sym.json()[:5], indent=2))

print("\n--- 4. Error Handling Check ---")
print("Calling POST /predict with empty list...")
resp_err = requests.post(f"{BASE_URL}/predict", json={"symptoms": []})
print(f"Status Code: {resp_err.status_code}")
print(f"Response: {resp_err.text}")
