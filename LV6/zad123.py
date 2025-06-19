import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn import datasets

def generate_data(n_samples, flagc):
    if flagc == 1:
        random_state = 365
        X, y = datasets.make_blobs(n_samples=n_samples, random_state=random_state)
    elif flagc == 2:
        random_state = 148
        X, y = datasets.make_blobs(n_samples=n_samples, random_state=random_state)
        transformation = [[0.60834549, -0.63667341], [-0.40887718, 0.85253229]]
        X = np.dot(X, transformation)
    elif flagc == 3:
        random_state = 148
        X, y = datasets.make_blobs(n_samples=n_samples,
                                   centers=4,
                                   cluster_std=[1.0, 2.5, 0.5, 3.0],
                                   random_state=random_state)
    elif flagc == 4:
        X, y = datasets.make_circles(n_samples=n_samples, factor=.5, noise=.05)
    elif flagc == 5:
        X, y = datasets.make_moons(n_samples=n_samples, noise=.05)
    else:
        X = []
    return X

n_samples = 500
flagc = 3
X = generate_data(n_samples, flagc)

k = 3
kmeans = KMeans(n_clusters=k, n_init=10)
labels = kmeans.fit_predict(X)
centers = kmeans.cluster_centers_

plt.figure(figsize=(8, 5))
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='tab10', s=30)
plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='x', s=100)
plt.title(f"KMeans klaster (k = {k})")
plt.xlabel("X")
plt.ylabel("Y")
plt.grid()
plt.show()

inertias = []
k_values = range(1, 21)
for i in k_values:
    km = KMeans(n_clusters=i, n_init=10)
    km.fit(X)
    inertias.append(km.inertia_)

plt.figure(figsize=(8, 4))
plt.plot(k_values, inertias, marker='o')
plt.xlabel("k")
plt.ylabel("Inertia")
plt.title("Elbow metoda")
plt.grid()
plt.xticks(k_values)
plt.show()

methods = ['single', 'complete', 'average', 'ward']

for method in methods:
    plt.figure(figsize=(10, 4))
    Z = linkage(X, method=method)
    dendrogram(Z, truncate_mode="level", p=10)
    plt.title(f"Dendrogram - metoda: {method}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.tight_layout()
    plt.show()
