import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "adult.csv"

df = pd.read_csv(DATA_DIR)
df = df.drop_duplicates().copy()
result = []
df["income"] = df["income"].astype(str).str.strip()
X = df.drop(columns="income")
y = (df["income"] == ">50K").astype(int)
X_train,X_test,y_train,y_test = train_test_split(X,y,stratify=y,random_state=42,test_size=0.2)
numeric_columns = X_train.select_dtypes(include="number").columns.to_list()
cat_columns = X_train.select_dtypes(exclude = "number").columns.to_list()

numeric_preprocessing = Pipeline(steps=[
    ("impute",SimpleImputer(strategy="mean")),
    ("scaler",StandardScaler())
])
cat_preprocessing = Pipeline(steps=[
    ("impute",SimpleImputer(strategy="constant",fill_value="missing")),
    ("scaler",OneHotEncoder(handle_unknown="ignore",sparse_output=False))
])
preprocessing = ColumnTransformer(transformers=[
    ("numeric",numeric_preprocessing,numeric_columns),
    ("cat",cat_preprocessing,cat_columns)
])
models = {
    "LogisticRegression":LogisticRegression(max_iter=1000,random_state=42),
    "randomForest":RandomForestClassifier(n_estimators=200,n_jobs=1,random_state=42),
    "GradientBoost":GradientBoostingClassifier(random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "XGB":XGBClassifier(n_estimators = 300,learning_rate = 0.1,max_depth = 4,subsample = 0.9,colsample_bytree = 0.9,eval_metric="logloss",random_state = 42)
}
for model,classifier in models.items():
    pipe = Pipeline(steps=[
        ("preprocessing",preprocessing),
        ("model",classifier)
    ])

    pipe.fit(X_train,y_train)

    y_pred = pipe.predict(X_test)
    print(f"model Name: {model}")
    accuracy = accuracy_score(y_test,y_pred)
    print("accuracy:\n",accuracy)
    confusionMatrix = confusion_matrix(y_test,y_pred)
    print("confusion matrix:\n",confusionMatrix)
    classificationReport = classification_report(y_test,y_pred,target_names=["<=50K",">50K"],output_dict=True,zero_division=0)
    print("classification report:\n",classificationReport)
    result.append({
        "name":model,
        "accuracy":accuracy,
        ">50K_precision":classificationReport[">50K"]["precision"],
        "<=50K_precision":classificationReport["<=50K"]["precision"],
        ">50K_recall":classificationReport[">50K"]["recall"],
        "<=50K_recall":classificationReport["<=50K"]["recall"],
        ">50K_f1":classificationReport[">50K"]["f1-score"],
        "<=50K_f1":classificationReport["<=50K"]["f1-score"]
    })
result_pd = pd.DataFrame(result)
result_pd = result_pd.sort_values(">50K_f1",ascending=False)
print("\n" + "=" *60)
print("Final model comparison table")
print("="*60)
print(result_pd)