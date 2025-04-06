import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report

# Load data from Tutorial 4 folder
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

# Models to train
models = {
    "SVM": SVC(),
    "SGD": SGDClassifier(),
    "RandomForest": RandomForestClassifier(),
    "MLP": MLPClassifier(max_iter=1000)
}

# Store classification reports
report_output = ""

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred)
    report_output += f"\n\n{name} Classification Report:\n"
    report_output += report

# Save to file
with open("Tutorial 4/classification_reports.txt", "w") as f:
    f.write(report_output)

# Print where the file is saved
print("Classification reports saved to 'Tutorial 4/classification_reports.txt'")
