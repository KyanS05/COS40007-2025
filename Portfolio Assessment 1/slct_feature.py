import pandas as pd

# Load engineered dataset
features_file_path = "Portfolio 1/features_powerplant.csv"
df_features = pd.read_csv(features_file_path)

# Select only most relevant features
selected_features = ["AT", "V", "RH", "PE_category"]
df_selected = df_features[selected_features]

# Save
df_selected.to_csv("Portfolio 1/selected_feature_powerplant.csv", index=False)

print("✅ Selected feature dataset saved as selected_feature_powerplant.csv")
