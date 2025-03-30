import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score
from sklearn.metrics import accuracy_score

# load shuffled data
data = pd.read_csv('Tutorial 3/all_data.csv')
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# train/test split (same as before)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# define parameter grid for rbf kernel
param_grid = {
    'C': [0.1, 1, 10],
    'gamma': [1, 0.1, 0.01],
    'kernel': ['rbf']
}

# grid search with 5-fold CV on training set
grid = GridSearchCV(SVC(), param_grid, refit=True, cv=5)
grid.fit(X_train, y_train)

# best model found
best_clf = grid.best_estimator_

# evaluate with train/test split
y_pred_tuned = best_clf.predict(X_test)
acc_split_tuned = accuracy_score(y_test, y_pred_tuned)

# evaluate with 10-fold CV on full dataset
scores_tuned = cross_val_score(best_clf, X, y, cv=10)
acc_cv_tuned = scores_tuned.mean()

# save result to file
results_tuned = pd.DataFrame({
    'model': ['svm_tuned'],
    'train_test_accuracy': [acc_split_tuned],
    'cross-validation scores': [scores_tuned],
    'average_accuracy': [acc_cv_tuned],
    'best_params': [grid.best_params_]
})
results_tuned.to_csv('Tutorial 3/tut3.3_result.csv', index=False)