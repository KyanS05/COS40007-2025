import pandas as pd
import matplotlib.pyplot as plt
import pickle
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.model_selection import train_test_split

# load training data
df = pd.read_csv("Portfolio 3/step1_train_cleaned.csv")

# extract only SP columns
sp_cols = [col for col in df.columns if "SP" in col and col != "Class"]
X = df[sp_cols]
y = df["Class"]

# split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42, test_size=0.3)

# train decision tree
tree = DecisionTreeClassifier(max_depth=5, random_state=42)
tree.fit(X_train, y_train)

# plot tree
plt.figure(figsize=(16, 10))
plot_tree(tree, feature_names=sp_cols, class_names=["0", "1", "2"], filled=True)
plt.title("Decision Tree with SP Features Only")
plt.show()

# export rules
rules = export_text(tree, feature_names=sp_cols)
with open("Portfolio 3/step4_rules.txt", "w") as f:
    f.write(rules)

print("Tree rules saved to Portfolio 3/step4_rules.txt")
