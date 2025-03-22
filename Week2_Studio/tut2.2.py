import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load converted dataset
file_path = "Tutorial 2/converted_concrete.csv"
df = pd.read_csv(file_path)

# Check if strength_category exists
if "strength_category" not in df.columns:
    print("Recalculating strength_category since missing.")
    def categorize_strength(value):
        if value < 20:
            return 1
        elif 20 <= value < 30:
            return 2
        elif 30 <= value < 40:
            return 3
        elif 40 <= value < 50:
            return 4
        else:
            return 5
    df["strength_category"] = df["strength"].apply(categorize_strength)

# Simplify the age feature
unique_ages = df["age"].unique()
age_mapping = {age: idx+1 for idx, age in enumerate(sorted(unique_ages))}
df["age_category"] = df["age"].map(age_mapping)

# Normalize seven features except age
features_normalize = ["cement", "slag", "ash", "water", "superplastic", "coarseagg", "fineagg"]
scaler = MinMaxScaler()
df_normalized = df.copy()
df_normalized[features_normalize] = scaler.fit_transform(df[features_normalize])

# Save normalized
df_normalized.to_csv("Tutorial 2/normalised_concrete.csv", index=False)

# Create composite features with covariance
df_composite = df_normalized.copy()
df_composite["cement_slag"] = df_composite["cement"] * df_composite["slag"]
df_composite["cement_ash"] = df_composite["cement"] * df_composite["ash"]
df_composite["water_fineagg"] = df_composite["water"] * df_composite["fineagg"]
df_composite["ash_superplastic"] = df_composite["ash"] + df_composite["superplastic"]

# Keep only required columns
final_features = ["age_category"] + features_normalize + ["cement_slag", "cement_ash", "water_fineagg", "ash_superplastic", "strength_category"]
df_composite = df_composite[final_features]

# Save the final feature-engineered dataset
df_composite.to_csv("Tutorial 2/features_concrete.csv", index=False)

print(f"Normalized dataset saved as normalised_concrete.csv")
print(f"Feature-engineered dataset saved as features_concrete.csv")
