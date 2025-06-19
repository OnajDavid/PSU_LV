import matplotlib.image as mpimg
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
image = mpimg.imread('example.png')
w, h, d = image.shape
X = image.reshape((-1, 3))

kmeans = KMeans(n_clusters=10, n_init=10)
kmeans.fit(X)

values = kmeans.cluster_centers_.astype('uint8')
labels = kmeans.labels_
image_compressed = values[labels]
image_compressed = image_compressed.reshape((w, h, d))

plt.figure()
plt.imshow(image)
plt.title('Originalna slika')
plt.axis('off')

plt.figure()
plt.imshow(image_compressed)
plt.title('10 klastera)')
plt.axis('off')
plt.show()
