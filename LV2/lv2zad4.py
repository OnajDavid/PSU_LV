import numpy as np
import matplotlib.pyplot as plt
dimension = 50
num_hoz = 5
num_ver = 4
ones = np.ones([dimension,dimension])
zeros = np.zeros([dimension,dimension])
img = np.hstack(ones)
plt.figure()
plt.imshow(img, cmap='gray', vmin=0, vmax=255)
plt.show()