"""One pixel camera simulation"""

import math
import json

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


def hadamard(size,idnum):
    """Return a hadamard matrix"""
    # size is 2 power to order
    order = int(math.log(size, 2))
    matrix_code = base_convert(idnum, 4)
    padding = np.zeros(int(math.log(size**2,4)) - len(matrix_code), dtype="int")
    vector = np.concatenate((padding, matrix_code))
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
        hadamard_matrix = hadamard(size,i)
        matrix_vector.append(hadamard_matrix)
    return matrix_vector


#TEST_IMAGE = scipy.misc.face()
TEST_IMAGE = plt.imread("/Users/eloymorenogarcia/Desktop/R.jpg")
TEST_IMAGE = TEST_IMAGE[:, :, 1]
TEST_IMAGE = scipy.misc.imresize(TEST_IMAGE, [256, 256])
x,y = TEST_IMAGE.shape
TEST_IMAGE = TEST_IMAGE.astype("float64")

nn = x*y
M = int((x*y)*0.1)
HADAMARD_MASKS = generate_hadamard(TEST_IMAGE.shape[0], nn)
np.random.shuffle(HADAMARD_MASKS)

result = np.ones(TEST_IMAGE.shape)
for i in range(0, M):
    mask = HADAMARD_MASKS[i] == 1
    mask = mask * 1
    masked = TEST_IMAGE * HADAMARD_MASKS[i]
    intensity = masked.sum(dtype="float64")
    result += intensity * HADAMARD_MASKS[i]

result = skimage.exposure.rescale_intensity(result, out_range=(0, 255))

print ssim(result, TEST_IMAGE)

fig = plt.figure()
fig.add_subplot(1, 2, 1)
plt.gray()
plt.imshow(TEST_IMAGE)
fig.add_subplot(1, 2, 2)
plt.imshow(result)
plt.show()