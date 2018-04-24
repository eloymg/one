"""One pixel camera simulation"""

import math
import random

import matplotlib.pyplot as plt
import numpy as np
import scipy.misc
import skimage.exposure
from skimage.measure import compare_ssim as ssim



def base_convert(number, base):
    """Convert number to a numerical base"""
    result = []
    if number == 0:
        return [0]
    while number > 0:
        result.insert(0, number % base)
        number = number // base
    return result


def hadamard(size, vector):
    """Return a hadamard matrix"""
    # size is 2 power to order
    order = int(math.log(size, 2))
    vector = vector[::-1]
    v_m = [[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]]
    h_m = np.array([[1]])

    for i in xrange(0, order):
        h_m1 = np.concatenate((v_m[vector[i]][0] * h_m,
                               v_m[vector[i]][2] * h_m))
        h_m2 = np.concatenate((v_m[vector[i]][1] * h_m,
                               v_m[vector[i]][3] * h_m))
        h_m = np.concatenate((h_m1, h_m2), 1)

    return h_m


def generate_hadamard(size, number):
    """Generate a n hadamard matrix """
    matrix_vector = []
    for i in xrange(0, number):
        matrix_code = base_convert(i, 4)
        padding = np.zeros(7 - len(matrix_code), dtype="int")
        matrix_code = np.concatenate((padding, matrix_code))
        hadamard_matrix = hadamard(size, matrix_code)
        matrix_vector.append(hadamard_matrix)
    return matrix_vector


TEST_IMAGE = scipy.misc.face()
TEST_IMAGE = TEST_IMAGE[:, :, 0]
TEST_IMAGE = scipy.misc.imresize(TEST_IMAGE, [128, 128])
TEST_IMAGE = TEST_IMAGE.astype("float64")

nn = 4**7

HADAMARD_MASKS = generate_hadamard(TEST_IMAGE.shape[0], nn)
np.random.shuffle(HADAMARD_MASKS)
ssim_vector = []
M_vector = []
for M in range(0,(4**7)*5,(4**7)*5/20):
    
    result = np.ones(TEST_IMAGE.shape)
    for i in range(0, M):
        #mask = HADAMARD_MASKS[i]
        index = np.random.random_integers(0,nn-1)
        mask = HADAMARD_MASKS[index]
        mask = mask == 1
        
        maskP = mask * 1
        maskedP = TEST_IMAGE * maskP
        maskN = np.logical_not(mask)*1
        maskedN = TEST_IMAGE * maskN
        intensity = maskedP.sum(dtype="float64")-maskedN.sum(dtype="float64")
        pixels = mask.sum(dtype="float64")
        result += intensity * mask

    result = skimage.exposure.rescale_intensity(result, out_range=(0, 255))
    M_vector.append(M)
    ssim_result = ssim(result, TEST_IMAGE)
    ssim_vector.append(ssim_result)
    print M,",",ssim_result

plt.plot(M_vector,ssim_vector)
plt.show()

