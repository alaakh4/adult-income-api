from fastapi import FastAPI
from api.model_service import prediction_income
from api.schemas import AdultIncomeInput,PredictionResponse

app = FastAPI(
    title="Adult Income Prediction Income",
    version="1.0.0"
)

@app.get("/")
def root():
    return{
        "msg":"Adult Income Prediction is running"
    }

@app.get("/health")
def health():
    return{
        "status":"ok",
        "model":"adult_income_v1"
    }
@app.post("/predict",response_model=PredictionResponse)
def predict(data: AdultIncomeInput):
    return prediction_income(data)
@app.get("/model-info")
def model_info():
    return{
        "model_name":"XGBoost",
        "model_version":"1.0.0",
        "problem_type":"Binary classification",
        "target":{
            "0":"<=50k",
            "1":">50k"
        },
        "features_count":14,
        "uses_pipeline":True
    }