from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from imblearn.combine import SMOTETomek
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load your data from Tutorial 4
w1 = pd.read_csv("Tutorial 4/w1.csv")
w2 = pd.read_csv("Tutorial 4/w2.csv")
w3 = pd.read_csv("Tutorial 4/w3.csv")

# Combine the datasets
df = pd.concat([w1, w2, w3], ignore_index=True)

# Split into features and labels
X = df.drop("class", axis=1)
y = df["class"]

# Balance the dataset
smt = SMOTETomek(random_state=42)
X_resampled, y_resampled = smt.fit_resample(X, y)

# Train-test split (on balanced data)
X_train_bal, X_test_bal, y_train_bal, y_test_bal = train_test_split(
    X_resampled, y_resampled, test_size=0.3, random_state=42, stratify=y_resampled
)

# Retrain RandomForest
model_rf_bal = RandomForestClassifier(random_state=42)
model_rf_bal.fit(X_train_bal, y_train_bal)
y_pred_bal = model_rf_bal.predict(X_test_bal)

# Classification report
print("RandomForest After Balancing:")
print(classification_report(y_test_bal, y_pred_bal))

# Confusion matrix
cm_bal = confusion_matrix(y_test_bal, y_pred_bal)
disp = ConfusionMatrixDisplay(confusion_matrix=cm_bal, display_labels=[0, 1, 2])
disp.plot()
plt.title("RandomForest Confusion Matrix (After Balancing)")
plt.show()
