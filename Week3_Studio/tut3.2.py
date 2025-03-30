import pandas as pd
from sklearn.model_selection import train_test_split # 30/70 test
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

## load data and split x and y
# load shuffled data
data = pd.read_csv('Tutorial 3/all_data.csv')

# separate features and labels
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

## train/test split and accuracy
# 70/30 split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# train svm
clf = SVC()
clf.fit(X_train, y_train)

# predict and evaluate
y_pred = clf.predict(X_test)
acc_split = accuracy_score(y_test, y_pred)
print("train/test accuracy:", acc_split)

## 10-fold cross-validation
# use same clf (SVC)
scores = cross_val_score(SVC(), X, y, cv=10)
print("cross-validation scores:", scores)
print("average cv accuracy:", scores.mean())

# save results
results_df = pd.DataFrame({
    'model': ['svm'],
    'train_test_accuracy': [acc_split],
    'cross-validation scores:': [scores],
    'average_accuracy': [scores.mean()]
})
results_df.to_csv('Tutorial 3/tut3.2_result.csv', index=False)