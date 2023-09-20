import numpy as np
from skimage import data, color, filters, segmentation
from skimage.util import img_as_float32, img_as_float64
from skimage.segmentation import slic as sk_slic
from skimage.segmentation import mark_boundaries
from cuda_slic.slic import slic as cuda_slic
import matplotlib.pyplot as plt
import time
from fast_slic.avx2 import SlicAvx2
def plot_images(images, cols=2, ax_size=5, titles=None):
    rows = (len(images)+cols-1)//cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols*ax_size, rows*ax_size))
    axes = axes.flatten()
    for ax in axes:
        ax.axis('off')
    for i in range(len(images)):
        axes[i].imshow(images[i])
        if titles:
            axes[i].set_title(titles[i], size=32)
    plt.tight_layout()



astro = data.astronaut()
# labels = cuda_slic(astro,n_segments=10, multichannel=True, compactness = 0.1)
# marked_astro = mark_boundaries(astro, labels)
# plot_images([astro, marked_astro])
# plt.show()

# # 2D RGB image
# img = data.astronaut() 
# labels = slic(img, n_segments=100, compactness=10)

# # 3D gray scale
# vol = data.binary_blobs(length=50, n_dim=3, seed=2)
# labels = cuda_slic(vol, n_segments=2, multichannel=False, compactness=0.1)
# # marked_vol = mark_boundaries(vol,labels)
# plot_images([vol[21], labels[21]])
# # # 3D multi-channel
# # volume with dimentions (z, y, x, c)
# # or video with dimentions (t, y, x, c)
# vol = data.binary_blobs(length=33, n_dim=4, seed=2)
# labels = slic(vol, n_segments=100, multichannel=True, compactness=1)
# from skimage import data
# from cuda_slic.slic import slic as cuda_slic
# from skimage.segmentation import slic as skimage_slic

# blob = data.binary_blobs(length=500, n_dim=3, seed=2)
# n_segments = 500**3/5**3 # super pixel shape = (5,5,5)
# plt.show()
start = time.time()
# labels = sk_slic(astro, n_segments=100, max_num_iter=5)
# marked_astro = mark_boundaries(astro, labels)

# labels = cuda_slic(astro,n_segments=100,max_iter =5)
slic = SlicAvx2(num_components=200, compactness=10,min_size_factor=0)
assignment = slic.iterate(astro)
end = time.time()
print("The time of execution of above program is :",
      (end-start) * 10**3, "ms")
marked_astro = mark_boundaries(astro, assignment)
plot_images([astro, marked_astro])
plt.show()
# import pdb;pdb.set_trace()

# # 2min 28s ± 0 ns per loop (mean ± std. dev. of 1 run, 1 loop each)

# timeit -r1 -n1 cuda_slic(blob, n_segments=n_segments, multichannel=False, max_iter=5)
# 13.1 s ± 0 ns per loop (mean ± std. dev. of 1 run, 1 loop each)