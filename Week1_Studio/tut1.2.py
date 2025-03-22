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

# remove duplicates
df = df.drop_duplicates()

# rename columns for clarity (if needed)
df.columns = df.columns.str.strip()

# confirm the changes
print("\nDataset after cleaning:")
print(df.info())
