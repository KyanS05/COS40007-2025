import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# load data
df = pd.read_csv('Portfolio 2/step3_features.csv')
X = df.iloc[:, :-1]
y = df['class']

# 70/30 split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# train baseline SVM
clf = SVC()
clf.fit(X_train, y_train)

# predict
y_pred = clf.predict(X_test)
acc_split = accuracy_score(y_test, y_pred)

# 10-fold CV
cv_scores = cross_val_score(SVC(), X, y, cv=10)
acc_cv = cv_scores.mean()

# save
results = pd.DataFrame({
    'model': ['svm_baseline'],
    'train_test_accuracy': [acc_split],
    'cross_validation_accuracy': [acc_cv]
})
results.to_csv('Portfolio 2/step4.1_baseline.csv', index=False)
print("step 4.1 done: saved as step4.1_baseline.csv")
