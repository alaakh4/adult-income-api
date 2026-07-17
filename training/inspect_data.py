import pandas as pd
from pathlib import Path

Base_Dir = Path(__file__).resolve().parents[1]
Data_Dir = Base_Dir / "data" / "adult.csv"

df = pd.read_csv(Data_Dir)

print("shape: ")
print(df.shape)
print("colmun: ")
print(df.columns.tolist())
print("First 5 rows")
print(df.head())
print("missing colmuns")
print(df.isnull().sum())
print("Data types")
print(df.dtypes)
print("Target Values")
print(df["income"].value_counts())
print("Target Precentege")
print(df["income"].value_counts(normalize=True))

features = df.drop(columns="income")

numeric_columns = features.select_dtypes(include="number").columns.tolist()
cat_columns = features.select_dtypes(exclude="number").columns.tolist()

print("numeric columns")
print(numeric_columns)
print("cat columns")
print(cat_columns)
print("length of num columns", len(numeric_columns))
print("length of cat columns", len(cat_columns))

for column in cat_columns:
    print(f"\n{column} : {df[column].nunique()} unique values")
    print(df[column].value_counts(dropna=False).head(10))
    print(df[column].isna().sum())

print("numeric summary: ", df[numeric_columns].describe().T)
print("nan values in numeric: ", df[numeric_columns].isnull().sum())

print("capital gain more than 0:")
print((df["capital-gain"] > 0).sum())
print("capital loss more than 0:")
print((df["capital-loss"] > 0).sum())
print("capital gain percentage more than 0:")
print((df["capital-gain"] > 0).mean() * 100)
print("capital loss percentage more than 0:")
print((df["capital-loss"] > 0).mean() * 100)

df["high_income"] = (df["income"] == ">50K").astype(int)

for column in ["education","workclass","sex","occupation","race"]:
    print(f"\ncolumn name: {column}")
    summary = (
        df.groupby(column,dropna=False)["high_income"]
        .agg(["mean","count"])
        .sort_values("mean",ascending=False)
    )
    summary["high_income_percent"] = summary["mean"] * 100
    print(summary[["count","high_income_percent"]])
print(f"\n Numeric summary by income group: ")
summary_numeric = (
    df.groupby("income",dropna=False)[numeric_columns]
    .mean()
    .T
)
print(summary_numeric)
print("capital summary: ")
capital_summary = df.groupby("income").agg(
    capital_gain_nonzero = (
        "capital-gain",
        lambda values: (values > 0).mean() * 100
    ),
    capital_loss_nonzero = (
        "capital-loss",
        lambda values: (values > 0).mean() * 100
    ),
    capital_gain_median = ("capital-gain","median"),
    capital_loss_median = ("capital-loss","median")
)
print(capital_summary)
print(f"duplicated rows:\n{df.duplicated().sum()}")
df = df.drop_duplicates().copy()
print(df.shape)


