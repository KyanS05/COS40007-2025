import pandas as pd
import matplotlib.pyplot as plt
import pickle

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.feature_selection import mutual_info_classif

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier

# load data
df = pd.read_csv("Portfolio 3/step1_train_cleaned.csv")
X = df.drop("Class", axis=1)
y = df["Class"]

# split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# top 20 features
mi = mutual_info_classif(X_train, y_train)
top20 = pd.Series(mi, index=X.columns).sort_values(ascending=False).head(20).index.tolist()

# save top20
with open("Portfolio 3/top20_features.pkl", "wb") as f:
    pickle.dump(top20, f)

X_train = X_train[top20]
X_test = X_test[top20]

# models
models = {
    "SVM": SVC(),
    "SGD": SGDClassifier(),
    "RandomForest": RandomForestClassifier(),
    "MLP": MLPClassifier(max_iter=1000),
    "DecisionTree": DecisionTreeClassifier()
}

# train & evaluate
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(f"\n{name} report:\n")
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1, 2])
    disp.plot()
    plt.title(f"{name} Confusion Matrix")
    plt.show()

    results[name] = model

# save best model
best_model = results["RandomForest"]
with open("Portfolio 3/best_model.pkl", "wb") as f:
    pickle.dump(best_model, f)