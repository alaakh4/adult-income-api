import joblib
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = BASE_DIR / "models" / "adult_income_v1.pkl"
DATA_DIR = BASE_DIR / "data" / "adult.csv"

model = joblib.load(MODEL_DIR)
df = pd.read_csv(DATA_DIR)
df = df.drop_duplicates().copy()
df["income"] = df["income"].astype(str).str.strip()
X = df.drop(columns="income")
y = (df["income"] == ">50K").astype(int)
sample = X.iloc[[0]]
y_pred = model.predict(sample)
y_pred_prob = model.predict_proba(sample)

print(f"model predict: {y_pred}\n model predict prob: {y_pred_prob}\n real y: {y.iloc[0]}")