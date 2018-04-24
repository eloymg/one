import math
import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack as spfft
import scipy.ndimage as spimg
import pickle

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

nx=64
ny=64

HADAMARD_MASKS = generate_hadamard(64, 64*64)
np.random.shuffle(HADAMARD_MASKS)

A = np.kron(
    spfft.idct(np.identity(nx), norm='ortho', axis=0),
    spfft.idct(np.identity(ny), norm='ortho', axis=0)
    )
mask_vec =[]
mask_vec_raw =[]
for i in range(0,nx*ny):
    mask = HADAMARD_MASKS[i]
    mask_vec_raw.append(mask)
    mask_vec.append(mask.T.flatten())
mask_vec = np.expand_dims(mask_vec, axis=1)
A = np.dot(mask_vec,A)

file = open('dict_hadamard.txt', 'w')
pickle.dump(A, file)
file.close()
file = open('masks_hadamard.txt', 'w')
pickle.dump(mask_vec_raw, file)
file.close()