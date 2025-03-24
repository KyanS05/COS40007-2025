import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load from converted dataset
df = pd.read_csv("Portfolio 1/converted_powerplant.csv")

# Normalize original features
features = ["AT", "V", "AP", "RH"]
scaler = MinMaxScaler()
df_normalised = df.copy()
df_normalised[features] = scaler.fit_transform(df[features])

# Save normalized dataset (no composites)
df_normalised.to_csv("Portfolio 1/normalised_powerplant.csv", index=False)

print("✅ Normalised dataset saved as normalised_powerplant.csv")
