from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
import pandas as pd

# load data
data = pd.read_csv('Tutorial 3/all_data.csv')
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# apply PCA to reduce to 10 components
pca = PCA(n_components=10)
X_pca = pca.fit_transform(X)

# split data
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.3, random_state=1)

# use tuned SVM from activity 3
clf = SVC(C=10, gamma=0.01, kernel='rbf')
clf.fit(X_train, y_train)

# evaluate
y_pred = clf.predict(X_test)
acc_split = accuracy_score(y_test, y_pred)
scores = cross_val_score(clf, X_pca, y, cv=10)
acc_cv = scores.mean()

# save result
result = pd.DataFrame({
    'model': ['svm_pca'],
    'train_test_accuracy': [acc_split],
    'cross-validation scores': [scores],
    'average_accuracy': [acc_cv]
})
result.to_csv('Tutorial 3/tut3.5_result.csv', index=False)