"""
Main
"""

import math
import numpy as np

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
        # size is 2 power to order
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
