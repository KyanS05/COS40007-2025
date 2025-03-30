import pandas as pd

# load raw data
boning = pd.read_csv('Portfolio 2/Boning.csv')
slicing = pd.read_csv('Portfolio 2/Slicing.csv')

# correct column names
cols = ['Frame', 'Neck x', 'Neck y', 'Neck z', 'Head x', 'Head y', 'Head z']

# extract and label
boning = boning[cols].copy()
boning['class'] = 0

slicing = slicing[cols].copy()
slicing['class'] = 1

# combine
combined = pd.concat([boning, slicing], ignore_index=True)

# save
combined.to_csv('Portfolio 2/step1_combined.csv', index=False)
print("step 1 done: saved as step1_combined.csv")