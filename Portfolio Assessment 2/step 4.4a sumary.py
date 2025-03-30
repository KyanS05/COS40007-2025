import pandas as pd

# load all results
r1 = pd.read_csv('Portfolio 2/step4.1_baseline.csv')
r2 = pd.read_csv('Portfolio 2/step4.2_tuned.csv')
r3 = pd.read_csv('Portfolio 2/step4.3_top10.csv')
r4 = pd.read_csv('Portfolio 2/step4.4_pca10.csv')

# combine into one summary table
summary = pd.DataFrame({
    'model': [
        'svm_baseline',
        'svm_tuned',
        'svm_top10_features',
        'svm_pca_10'
    ],
    'train_test_accuracy': [
        r1['train_test_accuracy'][0],
        r2['train_test_accuracy'][0],
        r3['train_test_accuracy'][0],
        r4['train_test_accuracy'][0]
    ],
    'cross_validation_accuracy': [
        r1['cross_validation_accuracy'][0],
        r2['cross_validation_accuracy'][0],
        r3['cross_validation_accuracy'][0],
        r4['cross_validation_accuracy'][0]
    ]
})

# save
summary.to_csv('Portfolio 2/step4.4a_summary_svm.csv', index=False)
print("summary saved: step4.4a_summary_svm.csv")