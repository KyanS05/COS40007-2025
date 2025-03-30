from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
import pandas as pd

# load data
data = pd.read_csv('Tutorial 3/all_data.csv')
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# select top 100 features
selector = SelectKBest(score_func=f_classif, k=100)
X_selected = selector.fit_transform(X, y)

# split data
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.3, random_state=1)

# use best params from activity 3 (replace with actual values if needed)
clf = SVC(C=10, gamma=0.01, kernel='rbf')
clf.fit(X_train, y_train)

# evaluate
y_pred = clf.predict(X_test)
acc_split = accuracy_score(y_test, y_pred)
scores = cross_val_score(clf, X_selected, y, cv=10)
acc_cv = scores.mean()

# save result
result = pd.DataFrame({
    'model': ['svm_feature_selected'],
    'train_test_accuracy': [acc_split],
    'cross-validation scores': [scores],
    'average_accuracy': [acc_cv]
})
result.to_csv('Tutorial 3/tut3.4_result.csv', index=False)
