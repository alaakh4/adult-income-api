from pydantic import BaseModel,Field

class AdultIncomeInput(BaseModel):
    age: int = Field(...,ge=17,le=100)
    workclass: str
    fnlwgt: int = Field(...,ge=0)
    education: str
    education_num : int = Field(...,ge=1,le=16)
    marital_status : str
    occupation: str
    relationship: str
    race:str
    sex:str
    capital_gain: int = Field(...,ge=0)
    capital_loss: int = Field(...,ge=0)
    hours_per_week: int = Field(...,ge=1,le=100)
    native_country: str

class PredictionProbabilities(BaseModel):
    less_equal_50k:float
    greater_50k:float

class PredictionResponse(BaseModel):
    prediction:int
    label:str
    probabilities:PredictionProbabilities