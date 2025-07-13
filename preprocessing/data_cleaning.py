import pandas as pd

df=pd.read_csv("fetal_health.csv")

# data cleaning:

null_vals=df.isnull().sum()
print(null_vals)

# any null val in the entire dataset
null_val_any=df.isnull().any().sum()
print(null_val_any)

# scaling and normalization is done in the code 