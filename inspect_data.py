import pandas as pd
from sklearn.datasets import fetch_california_housing

# 1. GET THE DATA
print("Fetching data...")
data = fetch_california_housing()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['median_house_value'] = data.target

# 2. INSPECT IT (See what needs cleaning)
print("\n--- First 5 Rows ---")
print(df.head())

print("\n--- Missing Values? ---")
print(df.isnull().sum())