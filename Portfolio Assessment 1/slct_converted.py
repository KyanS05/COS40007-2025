import pandas as pd

# Load converted dataset
df = pd.read_csv("Portfolio 1/converted_powerplant.csv")

# Select same features as pp2.3, but raw values
df_selected = df[["AT", "V", "RH", "PE_category"]]

# Save to file
df_selected.to_csv("Portfolio 1/selected_converted_powerplant.csv", index=False)

print("✅ Selected raw dataset saved as selected_converted_powerplant.csv")