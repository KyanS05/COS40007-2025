import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

# Make selected_converted_concrete.csv without normalisation
converted_df = pd.read_csv("Tutorial 2/converted_concrete.csv")

# Ensure strength_category exists
if "strength_category" not in converted_df.columns:
    print("Strength category missing. Converting 'strength' to 'strength_category'.")
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
    converted_df["strength_category"] = converted_df["strength"].apply(categorize_strength)

selected_converted_df = converted_df[["cement", "water", "superplastic", "age", "strength_category"]]
selected_converted_df.to_csv("Tutorial 2/selected_converted_concrete.csv", index=False)
print("Saved selected_converted_concrete.csv without normalisation")

# Define file paths for datasets
file_paths = {
    "Model 1": "Tutorial 2/converted_concrete.csv",
    "Model 2": "Tutorial 2/normalised_concrete.csv",
    "Model 3": "Tutorial 2/features_concrete.csv",
    "Model 4": "Tutorial 2/selected_feature_concrete.csv",
    "Model 5": "Tutorial 2/selected_converted_concrete.csv",
}

# Initialize a dictionary to store accuracy results
accuracy_results = {}

# Loop through each dataset, train a model, and calculate accuracy
for model_name, file_path in file_paths.items():
    # Load dataset
    df = pd.read_csv(file_path)
    print(f"Processing {file_path}")
    print("Columns in dataset:", df.columns)
    
    # Ensure strength_category exists
    if "strength_category" not in df.columns:
        print(f"'strength_category' missing in {file_path}. Checking alternatives...")
        if "strength" in df.columns:  # Convert strength to category if needed
            print(f"Converting 'strength' to 'strength_category' in {file_path}")
            df["strength_category"] = df["strength"].apply(categorize_strength)
        else:
            print(f"Neither 'strength_category' nor 'strength' found in {file_path}. Skipping this dataset.")
            continue  # Skip this dataset if both are missing
    
    # Define feature columns and target variable
    X = df.drop(columns=["strength_category"])  # Features
    y = df["strength_category"]  # Target variable
    
    # Split dataset into training (70%) and testing (30%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
    
    # Train Decision Tree Classifier
    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    
    # Make predictions
    y_pred = clf.predict(X_test)
    
    # Calculate accuracy
    accuracy = metrics.accuracy_score(y_test, y_pred)
    accuracy_results[model_name] = accuracy
    print(f"{model_name} Accuracy: {accuracy}")

# Convert results to a DataFrame
df_accuracy = pd.DataFrame(list(accuracy_results.items()), columns=["Model", "Accuracy"])

# Save accuracy results to a CSV file
df_accuracy.to_csv("Tutorial 2/model_accuracy_results.csv", index=False)

print("Model accuracy results saved as Tutorial 2/model_accuracy_results.csv")
