import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.preprocessing import FunctionTransformer
from sklearn.utils import resample

# Load the Wine Quality dataset
df = pd.read_csv(r'C:\Users\negis\OneDrive\Desktop\DU\SEM4\Data Mining\Practicals\wine.csv')

# Display the first few rows of the dataset
print("Original dataset:")
print(df.head())

# Let's select relevant features for pre-processing
selected_features = ['fixed acidity', 'volatile acidity', 'sulphates', 'alcohol', 'quality']
wine_subset_df = df[selected_features]

# Standardization
scaler = StandardScaler()
wine_scaled = scaler.fit_transform(wine_subset_df)

# Normalization
min_max_scaler = MinMaxScaler()
wine_normalized = min_max_scaler.fit_transform(wine_subset_df)

# Transformation (log transformation)
wine_transformed = np.log1p(wine_subset_df)

# Aggregation (mean aggregation)
wine_aggregated = wine_subset_df.groupby('quality').mean()

# Discretization/Binarization
# Let's binarize 'quality' feature into two classes: low quality (<=5) and high quality (>5)
binarizer = KBinsDiscretizer(n_bins=2, encode='ordinal', strategy='uniform')
quality_binarized = binarizer.fit_transform(wine_subset_df['quality'].values.reshape(-1, 1))

# Sampling (downsampling)
# Let's downsample the majority class (quality <= 5) to balance the dataset
wine_downsampled = pd.concat([
    resample(wine_subset_df[wine_subset_df['quality'] > 5], replace=False, n_samples=sum(wine_subset_df['quality'] <= 5)),
    wine_subset_df[wine_subset_df['quality'] <= 5]
])

# Now let's print out the results of each pre-processing technique
print("Standardized Data:")
print(pd.DataFrame(wine_scaled, columns=selected_features).head())

print("\nNormalized Data:")
print(pd.DataFrame(wine_normalized, columns=selected_features).head())

print("\nTransformed Data (Log Transformation):")
print(wine_transformed.head())

print("\nAggregated Data (Mean Aggregation by Quality):")
print(wine_aggregated)

print("\nBinarized Quality:")
print(pd.DataFrame(quality_binarized, columns=['quality_binarized']).head())

print("\nDownsampled Data:")
print(wine_downsampled.head())
