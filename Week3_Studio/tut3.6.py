import pandas as pd

# load all result files
results_2 = pd.read_csv('Tutorial 3/tut3.2_result.csv')
results_3 = pd.read_csv('Tutorial 3/tut3.3_result.csv')
results_4 = pd.read_csv('Tutorial 3/tut3.4_result.csv')
results_5 = pd.read_csv('Tutorial 3/tut3.5_result.csv')

# build summary table
summary_table = pd.DataFrame({
    'model': ['original_features', 'svm_tuned', 'svm_feature_selected', 'svm_pca'],
    'train_test_accuracy': [
        results_2['train_test_accuracy'][0],
        results_3['train_test_accuracy'][0],
        results_4['train_test_accuracy'][0],
        results_5['train_test_accuracy'][0]
    ],
    'cross_validation_accuracy': [
        results_2['average_accuracy'][0],  # change made here
        results_3['average_accuracy'][0],
        results_4['average_accuracy'][0],
        results_5['average_accuracy'][0]
    ]
})

# save summary
summary_table.to_csv('Tutorial 3/tut3.6_summary_result.csv', index=False)
print("Summary saved.")
