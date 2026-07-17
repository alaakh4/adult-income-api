from pathlib import Path
import pandas as pd
import joblib

MODEL_DIR = Path(__file__).resolve().parents[1] / "models" / "adult_income_v1.pkl"
model = joblib.load(MODEL_DIR)
sample = pd.DataFrame([{
    "age": 40,
    "workclass": "Private",
    "fnlwgt": 200000,
    "education": "Bachelors",
    "education-num": 13,
    "marital-status": "Married-civ-spouse",
    "occupation": "Exec-managerial",
    "relationship": "Husband",
    "race": "White",
    "sex": "Male",
    "capital-gain": 0,
    "capital-loss": 0,
    "hours-per-week": 45,
    "native-country": "United-States"
}])
y_pred = model.predict(sample)
y_pred_prob = model.predict_proba(sample)

label = ">50K" if y_pred == 1 else "<=50K"

print(y_pred)
print(label)
print(y_pred_prob)
