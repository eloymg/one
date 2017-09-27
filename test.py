"""One pixel camera simulation"""

import math
import random

import matplotlib.pyplot as plt
import numpy as np
import scipy.misc
from skimage.measure import compare_ssim as ssim
from sklearn.linear_model import OrthogonalMatchingPursuit
import skimage.transform

TEST_IMAGE = scipy.misc.face()
TEST_IMAGE = TEST_IMAGE[:, :, 0]
TEST_IMAGE = scipy.misc.imresize(TEST_IMAGE, [32, 32])
TEST_IMAGE = TEST_IMAGE.astype("float64")

F_TEST_IMAGE = np.fft.fft2(TEST_IMAGE)
RF_TEST_IMAGE = np.fft.fft2(skimage.transform.rotate(TEST_IMAGE,180))
plt.imshow(np.abs(RF_TEST_IMAGE)**2)
plt.show()
plt.imshow(np.abs(F_TEST_IMAGE)**2)
plt.imshow(np.abs(RF_TEST_IMAGE)**2-np.abs(F_TEST_IMAGE)**2)
plt.show()


    

