import matplotlib.image as mpimg
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt



image = mpimg.imread('example_grayscale.png')

w, h = image.shape
X = image.reshape((-1, 1))

kmeans = KMeans(n_clusters=10, n_init=10)
kmeans.fit(X)

values = kmeans.cluster_centers_.squeeze()
labels = kmeans.labels_
image_compressed = np.choose(labels, values)
image_compressed.shape = (w, h)

plt.figure()
plt.imshow(image, cmap='gray')
plt.title('Originalna slika')
plt.axis('off')

plt.figure()
plt.imshow(image_compressed, cmap='gray')
plt.title('Kvantizirana slika (10 klastera)')
plt.axis('off')
plt.show()

original_bits = 8
compressed_bits = np.ceil(np.log2(10))
compression_ratio = compressed_bits / original_bits
print(f"Kompresija: {compression_ratio:.2f}x manja")
