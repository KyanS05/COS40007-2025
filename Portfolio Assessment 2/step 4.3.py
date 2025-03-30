import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score

# load features
df = pd.read_csv('Portfolio 2/step3_features.csv')
X = df.iloc[:, :-1]
y = df['class']

# select top 10 features
selector = SelectKBest(score_func=f_classif, k=10)
X_selected = selector.fit_transform(X, y)

# split
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.3, random_state=1)

# tuned SVM
clf = SVC(C=1, gamma=0.1, kernel='rbf')
clf.fit(X_train, y_train)

# evaluate
y_pred = clf.predict(X_test)
acc_split = accuracy_score(y_test, y_pred)

cv_scores = cross_val_score(clf, X_selected, y, cv=10)
acc_cv = cv_scores.mean()

# save
results = pd.DataFrame({
    'model': ['svm_top10_features'],
    'train_test_accuracy': [acc_split],
    'cross_validation_accuracy': [acc_cv]
})
results.to_csv('Portfolio 2/step4.3_top10.csv', index=False)
print("step 4.3 done: saved as step4.3_top10.csv")
