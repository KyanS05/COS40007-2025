import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

# load features
df = pd.read_csv('Portfolio 2/step3_features.csv')
X = df.iloc[:, :-1]
y = df['class']

# train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# models to try
models = {
    'sgd': SGDClassifier(),
    'random_forest': RandomForestClassifier(),
    'mlp': MLPClassifier(max_iter=300)
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

# save
pd.DataFrame(results).to_csv('Portfolio 2/step4.5_other_models.csv', index=False)
print("step 4.5 done: saved as step4.5_other_models.csv")