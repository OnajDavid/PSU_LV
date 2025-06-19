import numpy as np
import matplotlib.pyplot as plt
img = plt.imread("C:\\Users\\student\\Desktop\\lv2\\tiger.png")
img = img[:,:,0].copy()
img_mirror = np.fliplr(img)
img_rot = np.rot90(img, 3)

img[:,:] += 0.2

plt.figure()
plt.imshow(img, cmap="gray",vmin = 0.0,vmax=0.7)
plt.show()