import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from imblearn.combine import SMOTETomek

# Load and combine w1, w2, w3
w1 = pd.read_csv("Tutorial 4/w1.csv")
w2 = pd.read_csv("Tutorial 4/w2.csv")
w3 = pd.read_csv("Tutorial 4/w3.csv")
df = pd.concat([w1, w2, w3], ignore_index=True)

X = df.drop("class", axis=1)
y = df["class"]

# Balance with SMOTE + Tomek
smt = SMOTETomek(random_state=42)
X_resampled, y_resampled = smt.fit_resample(X, y)

# Train-test split (70/30)
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.3, random_state=42, stratify=y_resampled
)

# Train SVM
svm_model = SVC()
svm_model.fit(X_train, y_train)

# Save SVM model
with open("Tutorial 4/svm_model.pkl", "wb") as f:
    pickle.dump(svm_model, f)

# Train RandomForest
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)

# Save RandomForest model
with open("Tutorial 4/rand_fors_model.pkl", "wb") as f:
    pickle.dump(rf_model, f)

print("Both models saved to Tutorial 4/")
