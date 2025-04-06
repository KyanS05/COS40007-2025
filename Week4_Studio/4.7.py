import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.feature_selection import mutual_info_classif
import matplotlib.pyplot as plt
import pickle

# Load data
w1 = pd.read_csv("Tutorial 4/w1.csv")
w2 = pd.read_csv("Tutorial 4/w2.csv")
w3 = pd.read_csv("Tutorial 4/w3.csv")
df = pd.concat([w1, w2, w3], ignore_index=True)
X = df.drop("class", axis=1)
y = df["class"]

# Select top 10 features using mutual_info_classif
mi_scores = mutual_info_classif(X, y)
mi_series = pd.Series(mi_scores, index=X.columns).sort_values(ascending=False)
top_features = mi_series.head(10).index.tolist()
X_top = X[top_features]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_top, y, test_size=0.3, random_state=42, stratify=y)

# Train decision tree
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)

# Save the model
with open("Tutorial 4/decision_tree_model.pkl", "wb") as f:
    pickle.dump(dt_model, f)

# Plot the tree
plt.figure(figsize=(16, 8))
plot_tree(dt_model, feature_names=top_features, class_names=["0", "1", "2"], filled=True)
plt.title("Decision Tree Visualization")
plt.show()

# Print tree structure as text
tree_text = export_text(dt_model, feature_names=top_features)
print(tree_text)

# Save text version of tree to a file
with open("Tutorial 4/decision_tree_text.txt", "w") as f:
    f.write(tree_text)

print("Decision tree rules saved to decision_tree_text.txt")
