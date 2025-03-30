import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

# load data
data = pd.read_csv('Tutorial 3/all_data.csv')
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# models
models = {
    'SGD': SGDClassifier(),
    'RandomForest': RandomForestClassifier(),
    'MLP': MLPClassifier(max_iter=300)
}

results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc_split = accuracy_score(y_test, y_pred)
    acc_cv = cross_val_score(model, X, y, cv=10).mean()
    results.append({
        'model': name,
        'train_test_accuracy': acc_split,
        'cross_validation_accuracy': acc_cv
    })

# add svm as well
svm_base = pd.read_csv('Tutorial 3/tut3.2_result.csv')
results.insert(0, {
    'model': 'SVM',
    'train_test_accuracy': svm_base['train_test_accuracy'][0],
    'cross_validation_accuracy': svm_base['average_accuracy'][0]
})

# save
pd.DataFrame(results).to_csv('Tutorial 3/tut3.7_result.csv', index=False)