class Masks(object):
    """
    todo
    """

    def __init__(self,image_size):
        """Constructor""" 
        self.image_size = image_size

    def Hadamard(self,id_num):
        """
        Private Method
        Return a hadamard matrix
        """
        # size is 2 power to order
        order = int(math.log(self.image_size, 2))
        matrix_code = self.Base_convert(id_num, 4)
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

    def Base_convert(self,number, base):
        """Convert number to a numerical base"""
        result = []
        if number == 0:
            return [0]
        while number > 0:
            result.insert(0, number % base)
            number = number // base
        return result
    def Generate_hadamard(self,number):
        """Generate a n hadamard matrix vector"""
        matrix_vector = []
        for i in xrange(0, number):
            
            hadamard_matrix = hadamard(matrix_code)
            matrix_vector.append(hadamard_matrix)
        return matrix_vector