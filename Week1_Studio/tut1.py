import pandas as pd

file_path = "Folds5x2_pp.csv"
df = pd.read_csv(file_path)

print("\nFirst 5 rows of the Dataset:")
print(df.head(5))

print("\nDataset Info:")
print(df.info())


print("\nMissing Values:")
print(df.isnull().sum())


print("\nSummary Statistics:")
print(df.describe())
