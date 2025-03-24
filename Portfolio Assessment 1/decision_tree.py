import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

# File paths for each dataset version
file_paths = {
    "Model 1": "Portfolio 1/converted_powerplant.csv",
    "Model 2": "Portfolio 1/normalised_powerplant.csv",
    "Model 3": "Portfolio 1/features_powerplant.csv",
    "Model 4": "Portfolio 1/selected_feature_powerplant.csv",
    "Model 5": "Portfolio 1/selected_converted_powerplant.csv"
}

# Dictionary to store results
accuracy_results = {}

# Loop through each dataset
for model_name, file_path in file_paths.items():
    df = pd.read_csv(file_path)
    
    # Ensure label exists
    if "PE_category" not in df.columns:
        print(f"⚠️ Skipping {model_name} - Missing target label.")
        continue

    # Separate features and target
    X = df.drop(columns=["PE_category"])
    y = df["PE_category"]

    # Train-test split (70-30)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

    # Train Decision Tree model
    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)

    # Predict & evaluate
    y_pred = clf.predict(X_test)
    accuracy = metrics.accuracy_score(y_test, y_pred)
    accuracy_results[model_name] = accuracy

    print(f"{model_name} Accuracy: {accuracy:.4f}")

# Save results to CSV
df_accuracy = pd.DataFrame(list(accuracy_results.items()), columns=["Model", "Accuracy"])
df_accuracy.to_csv("Portfolio 1/model_accuracy_results.csv", index=False)

print("\n✅ Model comparison results saved to model_accuracy_results.csv")
