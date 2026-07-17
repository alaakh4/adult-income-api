from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split,RandomizedSearchCV
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "adult.csv"

df = pd.read_csv(DATA_DIR)
df = df.drop_duplicates().copy()
df["income"] = df["income"].astype(str).str.strip()
X = df.drop(columns="income")
y = (df["income"] == ">50K").astype(int)
X_train,X_test,y_train,y_test = train_test_split(X,y,stratify=y,random_state=42,test_size=0.2)
numeric_columns = X_train.select_dtypes(include="number").columns.to_list()
cat_columns = X_train.select_dtypes(exclude = "number").columns.to_list()
numeric_preprocessing = Pipeline(steps=[
    ("imput",SimpleImputer(strategy="median")),
    ("scaler",StandardScaler())]
)
cat_preprocessing = Pipeline(steps=[
    ("impute",SimpleImputer(strategy="constant",fill_value="missing")),
    ("encoder",OneHotEncoder(handle_unknown="ignore"))]
)
columnsTrans = ColumnTransformer(transformers=[
    ("numeric",numeric_preprocessing,numeric_columns),
    ("cat",cat_preprocessing,cat_columns)]
)
xgb_model = Pipeline(steps=[
    ("preprocessor",columnsTrans),
    ("model",XGBClassifier(
        n_jobs=-1,
        objective = "binary:logistic",
        eval_metric = "logloss",
        random_state = 42
    ))]
)
param_distribution = {
    "model__n_estimators": [200, 300, 500],
    "model__learning_rate": [0.03, 0.05, 0.1],
    "model__max_depth": [3, 4, 5],
    "model__subsample": [0.8, 0.9, 1.0],
    "model__colsample_bytree": [0.8, 0.9, 1.0],
    "model__reg_lambda": [1, 3, 5, 10]
}
grid = RandomizedSearchCV(estimator=xgb_model,param_distributions=param_distribution,cv=3,n_iter=20,random_state=42,verbose=2,n_jobs=1,scoring="f1")

grid.fit(X_train,y_train)

best_params = grid.best_params_
model = grid.best_estimator_
best_score = grid.best_score_
print("\nBest parameters:")
print(grid.best_params_)

print("\nBest cross-validation F1 score:")
print(grid.best_score_)

y_pred = model.predict(X_test)
print(accuracy_score(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))

