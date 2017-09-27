"""One pixel camera simulation"""

import math
import random

import matplotlib.pyplot as plt
import numpy as np
import scipy.misc
from skimage.measure import compare_ssim as ssim
from sklearn.linear_model import OrthogonalMatchingPursuit


TEST_IMAGE = scipy.misc.face()
TEST_IMAGE = TEST_IMAGE[:, :, 0]
TEST_IMAGE = scipy.misc.imresize(TEST_IMAGE, [128, 128])
TEST_IMAGE = TEST_IMAGE.astype("float64")

intensity_vector = []
masks_vector = []
result = np.ones(TEST_IMAGE.shape)
for i in range(0, 12000):
        #mask = HADAMARD_MASKS[i]
        mask = np.random.rand(128,128)<0.5
        mask = mask * 1
        masked = TEST_IMAGE * mask
        intensity = masked.sum(dtype="float64")
        pixels = mask.sum(dtype="float64")
        masks_vector.append(mask.flatten().tolist())
        intensity_vector.append(intensity)

omp = OrthogonalMatchingPursuit(n_nonzero_coefs=12000)
omp.fit(masks_vector,intensity_vector)
result = np.reshape(omp.coef_,[128,128])
print np.nonzero(result)
plt.gray()
plt.imshow(result)
plt.show()
    

