from xgboost import XGBClassifier
import pandas as pd
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
import joblib

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "adult.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "adult_income_v1.pkl"
MODEL_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA_DIR)
df = df.drop_duplicates().copy()
df["income"] = df["income"].astype(str).str.strip()
X = df.drop(columns="income")
y = (df["income"] == ">50K").astype(int)

numeric_columns = X.select_dtypes(include="number").columns.to_list()
cat_columns = X.select_dtypes(exclude="number").columns.to_list()
numeric_preprocessing = Pipeline(steps=[
    ("impute",SimpleImputer(strategy="median"))
    ,("scaler",StandardScaler())
])
cat_preprocessing = Pipeline(steps=[
    ("impute",SimpleImputer(strategy="constant",fill_value="missing")),
    ("encoder",OneHotEncoder(handle_unknown="ignore"))
])
columnTrans = ColumnTransformer(transformers=[
    ("numeric",numeric_preprocessing,numeric_columns),
    ("cat",cat_preprocessing,cat_columns)
])
pipe = Pipeline(steps=[
    ("preprocessing",columnTrans),
    ("model",XGBClassifier(n_jobs = -1 , random_state = 42 ,objective = "binary:logistic",eval_metric = "logloss",n_estimators = 500,max_depth = 5,learning_rate = 0.1,colsample_bytree = 0.9,subsample = 1,reg_lambda = 5))
])
pipe.fit(X,y)
joblib.dump(pipe,MODEL_PATH)
print(f"Model saved to: {MODEL_PATH}")
print("Training completed successfully.")
