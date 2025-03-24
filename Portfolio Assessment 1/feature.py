import pandas as pd
from sklearn.preprocessing import MinMaxScaler

file_path = "Portfolio 1/converted_powerplant.csv"
df = pd.read_csv(file_path)

# Normalize selected features
features_to_normalize = ["AT", "V", "AP", "RH"]
scaler = MinMaxScaler()
df_normalised = df.copy()
df_normalised[features_to_normalize] = scaler.fit_transform(df[features_to_normalize])

# Create composite features
df_normalised["AT_RH"] = df_normalised["AT"] * df_normalised["RH"]
df_normalised["V_AP"] = df_normalised["V"] + df_normalised["AP"]
df_normalised["AT_squared"] = df_normalised["AT"] ** 2
df_normalised["RH_inverse"] = 1 / (df_normalised["RH"] + 0.01)  # Prevent divide-by-zero

# Final column selection
final_features = features_to_normalize + ["AT_RH", "V_AP", "AT_squared", "RH_inverse", "PE_category"]
df_final = df_normalised[final_features]

# Save
df_final.to_csv("Portfolio 1/features_powerplant.csv", index=False)

print("✅ Normalized and feature-engineered dataset saved as features_powerplant.csv")
