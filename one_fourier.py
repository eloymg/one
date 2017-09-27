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
TEST_IMAGE = scipy.misc.imresize(TEST_IMAGE, [32, 32])
TEST_IMAGE = TEST_IMAGE.astype("float64")
F_TEST_IMAGE = (np.fft.fft2(TEST_IMAGE))
plt.imshow(np.imag(F_TEST_IMAGE)**2)
plt.show()
x,y = TEST_IMAGE.shape
intensity_vector = []
masks_vector = []
result = np.ones(TEST_IMAGE.shape)
for i in range(0, 1050):
        
        mask = np.random.rand(x,y)<0.5
        maskP = mask * 1
        maskedP = np.real(F_TEST_IMAGE) * maskP
        maskN = np.logical_not(mask)*-1
        maskedN = np.real(F_TEST_IMAGE) * maskN
        maskT = maskP+maskN
        intensity = maskedP.sum()+maskedN.sum()
        masks_vector.append(maskT.flatten())
        intensity_vector.append(intensity)

omp = OrthogonalMatchingPursuit()
omp.fit(masks_vector,intensity_vector)
f_result = np.reshape(omp.coef_,[x,y])
result = np.fft.ifft2(f_result)
plt.imshow(abs(f_result))
plt.show()
plt.figure()
plt.subplot(1,2,1)
plt.imshow(TEST_IMAGE,"gray")
plt.subplot(1,2,2)
plt.imshow(np.real(result),"gray")
plt.show()
    

