import pandas as pd
from scipy import stats
import numpy as np

# Load the dataset

df = pd.read_csv(r'C:\Users\negis\OneDrive\Desktop\DU\SEM4\Data Mining\Practicals\wine.csv')

print("\nThe original dataframe is :-")
print(df)

#Finding missing values
miss=df[df=="?"].count().sum()
print("Total missing values : ",miss)

miss=df[df=="?"].count()
print("Missing values in every column :-")
print(miss)

df=df.replace("?",np.nan)

# Handle missing values by imputing with mean
df=df.dropna()
print("\nDataframe sfter removing missing values is :-")
print(df)

# Detect outliers using Z-score
numeric_columns = df.select_dtypes(include=[np.number])
z_scores = np.abs(stats.zscore(numeric_columns))
threshold = 3
outlier_rows, outlier_columns = np.where(z_scores > threshold)

# Remove outliers
df = df[(np.abs(z_scores) < threshold).all(axis=1)]
print("\nDataframe after removing outliers :-")
print(df)

# Define validation rules
min_values = {
    'fixed acidity': 4,
    'volatile acidity': 0,
    'citric acid': 0,
    'residual sugar': 0,
    'chlorides': 0,
    'free sulfur dioxide': 0,
    'total sulfur dioxide': 0,
    'density': 0.98,
    'pH': 2.5,
    'sulphates': 0,
    'alcohol': 8
}

max_values = {
    'fixed acidity': 16,
    'volatile acidity': 1.5,
    'citric acid': 1,
    'residual sugar': 15,
    'chlorides': 0.6,
    'free sulfur dioxide': 100,
    'total sulfur dioxide': 300,
    'density': 1.003,
    'pH': 4,
    'sulphates': 2,
    'alcohol': 15
}


print("\nThe columns in the dataframe are :-")
print(df.columns)

# Apply validation rules
for column in df.columns:
    if column in min_values:
        df = df[(df[column] >= min_values[column]) & (df[column] <= max_values[column])]

# Reset index after filtering
df.reset_index(drop=True, inplace=True)

print("\n",df.head())
