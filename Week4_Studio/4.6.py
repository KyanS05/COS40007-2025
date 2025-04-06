import pandas as pd
import pickle

# Load w4 data
w4 = pd.read_csv("Tutorial 4/w4.csv")
X_w4 = w4.drop("class", axis=1)
y_w4 = w4["class"]

# --- SVM MODEL (svm_model.pkl) ---
model1 = pickle.load(open("Tutorial 4/svm_model.pkl", "rb"))

correct1 = 0
for i in range(len(X_w4)):
    x_i = X_w4.iloc[i:i+1]  # Keep it as DataFrame
    y_i = y_w4.iloc[i]
    y_pred = model1.predict(x_i)[0]
    if y_pred == y_i:
        correct1 += 1

accuracy1 = correct1 / len(y_w4)
print(f"SVM Model Accuracy on w4: {accuracy1:.2f}")

# --- RANDOM FOREST MODEL (rand_fors_model.pkl) ---
model2 = pickle.load(open("Tutorial 4/rand_fors_model.pkl", "rb"))

correct2 = 0
for i in range(len(X_w4)):
    x_i = X_w4.iloc[i:i+1]
    y_i = y_w4.iloc[i]
    y_pred = model2.predict(x_i)[0]
    if y_pred == y_i:
        correct2 += 1

accuracy2 = correct2 / len(y_w4)
print(f"RandomForest Model Accuracy on w4: {accuracy2:.2f}")
