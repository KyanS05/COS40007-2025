import pandas as pd

features_file_path = "Tutorial 2/features_concrete.csv"
df_features = pd.read_csv(features_file_path)

# Select only relevant features
selected_features = ["cement", "water", "superplastic", "age_category", "strength_category"]
df_selected = df_features[selected_features]

# Save new dataset
df_selected.to_csv("Tutorial 2/selected_feature_concrete.csv", index=False)

print(f"Selected feature dataset saved as selected_feature_concrete.csv")