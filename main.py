"""
Main
"""

import math
import numpy as np
import scipy.misc
import matplotlib.pyplot as plt
import cvxpy as cvx

class One(object):
    def __init__(self):

    def benchmark():

    def reconstruction(self, method='', image, samples)
        if method = '':

        

        elif method = 'fourier':

            A = np.kron(
            spfft.idct(np.identity(nx), norm='ortho', axis=0),
            spfft.idct(np.identity(ny), norm='ortho', axis=0)
            )
            m = Masks()
            mask_vec = m.generate_random(samples)
            mask_vec_raw =[]
            for i in range(0,nx*ny):
                mask_vec.append(mask_vec[i].T.flatten())
            mask_vec_formated = np.expand_dims(mask_vec, axis=1)
            A = np.dot(mask_vec_formated,A)

            for i in range(0,M):
                mask = mask_vec[i]
                masked = mask*image
                intensity = np.sum(masked)
                intensity_vec.append(intensity)

            # do L1 optimization
            AX = A[0:M,:]
            vx = cvx.Variable(nx * ny)
            objective = cvx.Minimize(cvx.norm(vx, 1))
            constraints = [AX*vx == intensity_vec]
            prob = cvx.Problem(objective, constraints)
            result = prob.solve()
            Xat2 = np.array(vx.value).squeeze()
            Xat = Xat2.reshape(nx, ny).T
            Xa = self.idct2(Xat)
            Xa = Xa.astype("float64")
    @staticmethod
    def idct2(x):
        return spfft.idct(spfft.idct(x.T, norm='ortho', axis=0).T, norm='ortho', axis=0)

class Image(object):
    def __init__(self):
        """
        Constructor
        """
    def return_image(self, size = 64, path = ''):
        """
        Return image
        """
        if path =='':
            r_image = self.__normalize_image(scipy.misc.face(), size)
        else:
            r_image = self.__normalize_image(plt.imread(path), size)
        return r_image
    @staticmethod
    def __normalize_image(image, size):
        n_image = image[:, :, 1]
        n_image = scipy.misc.imresize(n_image, [size, size])
        n_image = n_image.astype("float64")
        return n_image
class Masks(object):
    """
    todo
    """

    def __init__(self, image_size):
        """Constructor"""

        self.image_size = image_size

    def hadamard(self, id_num):
        """
        Private Method
        Return a hadamard matrix
        """
        order = int(math.log(self.image_size, 2))
        matrix_code = self.__base_convert(id_num, 4)
        padding = np.zeros(7 - len(matrix_code), dtype="int")
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

    def generate_hadamard(self, number):
        """Generate a n hadamard matrix vector"""
        matrix_vector = []
        for i in xrange(0, number):
            hadamard_matrix = self.hadamard(i)
            matrix_vector.append(hadamard_matrix)
        return matrix_vector

    def generate_random(self, number):
        """Generate a n random matrix vector"""
        matrix_vector = []
        for _ in xrange(0, number):
            random_matrix = (np.random.rand(self.image_size, self.image_size) < 0.5)*1
            matrix_vector.append(random_matrix)
        return matrix_vector

    @staticmethod
    def __base_convert(number, base):
        """Convert number to a numerical base"""
        result = []
        if number == 0:
            return [0]
        while number > 0:
            result.insert(0, number % base)
            number = number // base
        return result
