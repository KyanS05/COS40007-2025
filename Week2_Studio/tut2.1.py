import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
file_path = "Tutorial 2/concrete.csv"  # Ensure this file is in the same directory or provide the full path
df = pd.read_csv(file_path)

# Define a function to categorize strength values
def categorize_strength(value):
    if value < 20:
        return 1  # Very low
    elif 20 <= value < 30:
        return 2  # Low
    elif 30 <= value < 40:
        return 3  # Moderate
    elif 40 <= value < 50:
        return 4  # Strong
    else:
        return 5  # Very strong

# Apply the function to create a new categorical column
df["strength_category"] = df["strength"].apply(categorize_strength)

# Drop the original strength column
df = df.drop(columns=["strength"])

# Save the converted dataset
df.to_csv("Tutorial 2/converted_concrete.csv", index=False)

# Plot the distribution of the classes
plt.figure(figsize=(8, 5))
df["strength_category"].value_counts().sort_index().plot(kind="bar")
plt.xlabel("Strength Category")
plt.ylabel("Count")
plt.title("Distribution of Strength Categories")
plt.xticks(ticks=[0, 1, 2, 3, 4], labels=["Very Low", "Low", "Moderate", "Strong", "Very Strong"], rotation=45)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.show()

print(f"Converted dataset saved as converted_concrete.csv")
