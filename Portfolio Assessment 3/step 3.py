import pandas as pd
import pickle
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# load test set
df_test = pd.read_csv("Portfolio 3/step1_test_data.csv")

# load model
with open("Portfolio 3/best_model.pkl", "rb") as f:
    model = pickle.load(f)

# load feature list
with open("Portfolio 3/top20_features.pkl", "rb") as f:
    top20 = pickle.load(f)

# select features
X_test = df_test[top20]
y_test = df_test["Class"]

# predict and show
y_pred = model.predict(X_test)
print("AI Test Set Performance:")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1, 2])
disp.plot()
plt.title("Best Model on 1000-Row AI Test Set")
plt.show()