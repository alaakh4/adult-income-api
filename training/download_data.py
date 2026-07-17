from pathlib import Path
import pandas as pd
from sklearn.datasets import fetch_openml

Base_Dir = Path(__file__).resolve().parents[1]
Data_Dir = Base_Dir / "data"

Data_Dir.mkdir(exist_ok=True)
X,y = fetch_openml(name="adult",version=2,return_X_y=True,as_frame=True)

df = X.copy()
df["income"] = y

file_path = Data_Dir / "adult.csv"
df.to_csv(file_path,index = False)
print("Dataset saved to:", file_path)
print("Shape:", df.shape)

print("\nFirst 5 rows:")
print(df.head())

print("\nColumn types:")
print(df.dtypes)

print("\nTarget values:")
print(df["income"].value_counts())