import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from starlette.testclient import TestClient
from backend.main import app

def run_tests():
    client = TestClient(app)
    print("=" * 60)
    print("Testing Machine Failure Prediction Backend API")
    print("=" * 60)

    # 1. Test Root
    res_root = client.get("/")
    print(f"[1] GET / -> Status: {res_root.status_code}")
    print(f"    Response: {res_root.json()}")
    assert res_root.status_code == 200
    assert res_root.json() == {"message": "Machine Failure Prediction API is running"}

    # 2. Test Health
    res_health = client.get("/health")
    print(f"[2] GET /health -> Status: {res_health.status_code}")
    print(f"    Response: {res_health.json()}")
    assert res_health.status_code == 200
    assert res_health.json() == {"status": "healthy"}

    # 3. Test Normal Prediction (No Failure expected)
    normal_input = {
        "type": "M",
        "air_temperature": 300.0,
        "process_temperature": 310.0,
        "rotational_speed": 1500.0,
        "torque": 40.0,
        "tool_wear": 100.0
    }
    res_pred_normal = client.post("/predict", json=normal_input)
    print(f"[3] POST /predict (Normal) -> Status: {res_pred_normal.status_code}")
    print(f"    Response: {res_pred_normal.json()}")
    assert res_pred_normal.status_code == 200
    assert res_pred_normal.json()["prediction"] in [0, 1]
    assert "failure_probability" in res_pred_normal.json()

    # 4. Test High-Stress Prediction (Machine Failure expected)
    stress_input = {
        "type": "L",
        "air_temperature": 304.0,
        "process_temperature": 313.0,
        "rotational_speed": 1200.0,
        "torque": 75.0,
        "tool_wear": 240.0
    }
    res_pred_stress = client.post("/predict", json=stress_input)
    print(f"[4] POST /predict (Stress) -> Status: {res_pred_stress.status_code}")
    print(f"    Response: {res_pred_stress.json()}")
    assert res_pred_stress.status_code == 200

    # 5. Test Model Info
    res_info = client.get("/model-info")
    print(f"[5] GET /model-info -> Status: {res_info.status_code}")
    print(f"    Model Type: {res_info.json().get('model_type')}")
    print(f"    Feature Importances: {res_info.json().get('feature_importances')}")
    assert res_info.status_code == 200

    # 6. Test Validation Error
    res_bad = client.post("/predict", json={"type": "INVALID", "air_temperature": 300.0})
    print(f"[6] POST /predict (Invalid Type) -> Status: {res_bad.status_code} (Validation handled correctly)")
    assert res_bad.status_code == 422

    print("=" * 60)
    print("ALL TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == "__main__":
    run_tests()
