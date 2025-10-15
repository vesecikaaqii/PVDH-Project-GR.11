import pandas as pd

ds = pd.read_csv("fifa21 raw data v2.csv", low_memory=False)

# Quick check per kolona
print(ds.shape)
print(ds.columns)
ds.head()
