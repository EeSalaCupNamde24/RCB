import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import numpy as np

data=load_digits().data
pca=PCA(n_components=2)
df=pca.fit_transform(data)

kmeans = KMeans(n_clusters=10)
labels = kmeans.fit_predict(df)

for i in np.unique(labels):
    filtered_data = df[labels == 1]
    plt.scatter(filtered_data[:,0],filtered_data[:,1],label=f"Cluster {i}")

plt.legend()
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.title("K-Means Cluster for Digits Dataset")
plt.show()

