import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Import models
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

# Load data
w1 = pd.read_csv("Tutorial 4/w1.csv")
w2 = pd.read_csv("Tutorial 4/w2.csv")
w3 = pd.read_csv("Tutorial 4/w3.csv")

# Combine datasets
df = pd.concat([w1, w2, w3], ignore_index=True)
X = df.drop("class", axis=1)
y = df["class"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Re-train models to reuse (if needed)
models = {
    "SVM": SVC(),
    "SGD": SGDClassifier(),
    "RandomForest": RandomForestClassifier(),
    "MLP": MLPClassifier(max_iter=1000)
}

# Plot confusion matrices
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1, 2])
    disp.plot()
    plt.title(f"{name} Confusion Matrix")
    plt.show()
