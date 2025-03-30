import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score
from sklearn.metrics import accuracy_score

# load data
df = pd.read_csv('Portfolio 2/step3_features.csv')
X = df.iloc[:, :-1]
y = df['class']

# train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# define parameter grid
param_grid = {
    'C': [1, 10],
    'gamma': [0.1, 0.01],
    'kernel': ['rbf']
}

# grid search
grid = GridSearchCV(SVC(), param_grid, refit=True, cv=5)
grid.fit(X_train, y_train)

# best model
best_model = grid.best_estimator_

# evaluate train/test split
y_pred = best_model.predict(X_test)
acc_split = accuracy_score(y_test, y_pred)

# evaluate 10-fold cv
cv_scores = cross_val_score(best_model, X, y, cv=10)
acc_cv = cv_scores.mean()

# save
results = pd.DataFrame({
    'model': ['svm_tuned'],
    'train_test_accuracy': [acc_split],
    'cross_validation_accuracy': [acc_cv],
    'best_params': [str(grid.best_params_)]
})
results.to_csv('Portfolio 2/step4.2_tuned.csv', index=False)
print("step 4.2 done: saved as step4.2_tuned.csv")
