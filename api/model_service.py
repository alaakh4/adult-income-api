import joblib
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = BASE_DIR / "models" / "adult_income_v1.pkl"
model = joblib.load(MODEL_DIR)

def prediction_income(data):
    input_dict = data.model_dump()
    model_input = pd.DataFrame([{
        "age" : input_dict["age"],
        "workclass":input_dict["workclass"],
        "fnlwgt":input_dict["fnlwgt"],
        "education":input_dict["education"],
        "education-num":input_dict["education_num"],
        "marital-status":input_dict["marital_status"],
        "occupation":input_dict["occupation"],
        "relationship":input_dict["relationship"],
        "race":input_dict["race"],
        "sex":input_dict["sex"],
        "capital-gain":input_dict["capital_gain"],
        "capital-loss":input_dict["capital_loss"],
        "hours-per-week":input_dict["hours_per_week"],
        "native-country":input_dict["native_country"]
    }])
    prediction = model.predict(model_input)[0]
    predictionProba = model.predict_proba(model_input)[0]
    label = ">50k" if prediction == 1 else "<=50k"
    return{
        "prediction":int(prediction),
        "label":label,
        "probabilities":{
            "less_equal_50k": float(predictionProba[0]),
            "greater_50k":float(predictionProba[1])
        }
    }