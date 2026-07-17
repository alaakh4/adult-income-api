from fastapi.testclient import TestClient
from api.main import app
import pytest
client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["model"] == "adult_income_v1"

def test_model_info():
    response = client.get("model-info")
    assert response.status_code == 200
    data = response.json()
    target = data["target"]
    assert data["model_name"] == "XGBoost"
    assert data["model_version"] == "1.0.0"
    assert data["problem_type"] == "Binary classification"
    assert target["0"] == "<=50k"
    assert target["1"] == ">50k"
    assert data["features_count"] == 14
    assert data["uses_pipeline"] is True

def test_predict_valid_input():
    payload = {
        "age": 40,
        "workclass": "Private",
        "fnlwgt": 200000,
        "education": "Bachelors",
        "education_num": 13,
        "marital_status": "Married-civ-spouse",
        "occupation": "Exec-managerial",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital_gain": 0,
        "capital_loss": 0,
        "hours_per_week": 45,
        "native_country": "United-States"
    }
    response = client.post("/predict",json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] in [0, 1]
    assert data["label"] in ["<=50k", ">50k"]

    assert "probabilities" in data
    assert "less_equal_50k" in data["probabilities"]
    assert "greater_50k" in data["probabilities"]

    assert isinstance(data["probabilities"]["less_equal_50k"], float)
    assert isinstance(data["probabilities"]["greater_50k"], float)

def test_predict_invalid_age():
    payload = {
        "age": 120,
        "workclass": "Private",
        "fnlwgt": 200000,
        "education": "Bachelors",
        "education_num": 13,
        "marital_status": "Married-civ-spouse",
        "occupation": "Exec-managerial",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital_gain": 0,
        "capital_loss": 0,
        "hours_per_week": 45,
        "native_country": "United-States"
    }
    response = client.post("/predict",json=payload)
    assert response.status_code == 422

@pytest.mark.parametrize(
    "field,invalid_value",[
        ("hours_per_week",0),
        ("capital_gain",-100),
        ("education_num",20)
    ]
)
def test_predict_invalid_numeric_inputs(field, invalid_value):
    payload = {
        "age": 40,
        "workclass": "Private",
        "fnlwgt": 200000,
        "education": "Bachelors",
        "education_num": 13,
        "marital_status": "Married-civ-spouse",
        "occupation": "Exec-managerial",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital_gain": 0,
        "capital_loss": 0,
        "hours_per_week": 45,
        "native_country": "United-States"
    }
    payload[field] = invalid_value
    response = client.post("/predict",json=payload)
    assert response.status_code == 422

def test_handle_unseed_data():
    payload = {
        "age": 40,
        "workclass": "Private",
        "fnlwgt": 200000,
        "education": "Bachelors",
        "education_num": 13,
        "marital_status": "Married-civ-spouse",
        "occupation": "Exec-managerial",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital_gain": 0,
        "capital_loss": 0,
        "hours_per_week": 45,
        "native_country": "Egypt"
    }
    response = client.post("predict",json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] in [0,1]
    assert data["label"] in [">50k","<=50k"]
    probabilities = data["probabilities"]
    total_proba = (probabilities["less_equal_50k"] + probabilities["greater_50k"])
    assert (total_proba - 1) < 0.001