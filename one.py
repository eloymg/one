import math

import matplotlib.pyplot as plt
import numpy as np
import scipy.misc
import skimage.exposure
from skimage.measure import compare_ssim as ssim


def base_convert(i, base):
    result = []
    if i == 0:
        return [0]
    while i > 0:
        result.insert(0, i % base)
        i = i // base
    return result

def hadamard(size, vector):

    # size is 2 power to order
    order = int(math.log(size, 2))
    vector = vector[::-1]
    v_m = [[1,  1,  1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]]
    h_m = np.array([[1]])
    
    for i in xrange(0, order):
        h_m1 = np.concatenate((v_m[vector[i]][0]*h_m, v_m[vector[i]][2]*h_m))
        h_m2 = np.concatenate((v_m[vector[i]][1]*h_m, v_m[vector[i]][3]*h_m))
        h_m = np.concatenate((h_m1, h_m2), 1)
    
    return h_m
def generate_hadamard(n):
    matrix_vector = []
    for i in xrange(0, n):
        c = base_convert(i, 4) 
        padding = np.zeros(7-len(c), dtype="int")      
        c = np.concatenate((padding, c))      
        hadamard_matrix = hadamard(128, c)
       
        matrix_vector.append(hadamard_matrix)

    return matrix_vector

test_image = scipy.misc.face()
test_image = test_image[:, :, 0]
test_image = scipy.misc.imresize(test_image, [128, 128])
test_image = test_image.astype("float64")

nn=4**7
M=4**7
hadamard_masks = generate_hadamard(nn)
np.random.shuffle(hadamard_masks)

res = np.ones(test_image.shape)
num=1
for i in range(0, M):
    mask = hadamard_masks[i]==1
    mask = mask*1
    masked = test_image*hadamard_masks[i]
    intensity = masked.sum(dtype="float64")
    pixels=hadamard_masks[i].sum(dtype="float64")
    
    res += (((intensity))*hadamard_masks[i])
    num+=1

res = skimage.exposure.rescale_intensity(res, out_range=(0,  255))

print ssim(res, test_image)

fig = plt.figure()
fig.add_subplot(1, 2, 1)
plt.gray()
plt.imshow(test_image)
fig.add_subplot(1, 2, 2)
plt.imshow(res)
plt.show()
