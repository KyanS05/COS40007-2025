import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay


# Use the original unbalanced data
# (Assume X and y are already defined from earlier w1+w2+w3)
# Load original data
w1 = pd.read_csv("Tutorial 4/w1.csv")
w2 = pd.read_csv("Tutorial 4/w2.csv")
w3 = pd.read_csv("Tutorial 4/w3.csv")

df = pd.concat([w1, w2, w3], ignore_index=True)
X = df.drop("class", axis=1)
y = df["class"]

# Train-test split on unbalanced data again
X_train_wt, X_test_wt, y_train_wt, y_test_wt = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Train with class weights
weighted_rf = RandomForestClassifier(
    class_weight={0: 0.3, 1: 0.5, 2: 0.2}, random_state=42
)
weighted_rf.fit(X_train_wt, y_train_wt)
y_pred_wt = weighted_rf.predict(X_test_wt)

# Classification report
print("RandomForest with Class Weights:")
print(classification_report(y_test_wt, y_pred_wt))

# Confusion matrix
cm_wt = confusion_matrix(y_test_wt, y_pred_wt)
disp = ConfusionMatrixDisplay(confusion_matrix=cm_wt, display_labels=[0, 1, 2])
disp.plot()
plt.title("RandomForest Confusion Matrix (With Class Weights)")
plt.show()
