"""
Main
"""

import math
import socket

import matplotlib.pyplot as plt
import numpy as np
import scipy.fftpack as spfft
import scipy.misc
from Crypto import Random
from Crypto.Cipher import AES, ARC4
from Crypto.Hash import SHA
from Crypto.Util import Counter, number

import _thread


class Client(object):
    def __init__(self, cipher_mode='AES',IP="127.0.0.1",PORT="9999"):
        self.data = []
        self.cipher_mode = cipher_mode
        self.IP = IP
        self.PORT = PORT

    def handler(self, _):
        nonce_counter = 1001
        while True:
            if nonce_counter > 1000:
                if self.cipher_mode == 'AES':
                    cipher, nonce = self.__cipher_AES()
                elif self.cipher_mode == 'RC4':
                    cipher, nonce = self.__cipher_RC4()
                else:
                    print('Bad cipher mode')
                    break
                response = self.send(b'nonce:' + nonce)
                response = str(response, 'utf-8')
                if response == "timeout":
                    print('timeout!')
                elif response == "ACK":
                    nonce_counter = 0
                else:
                    print('bad data')

            if len(self.data) > 0:
                data = self.data[0]
                response = self.send(b'data:' + cipher.encrypt(data))
                if response == "timeout":
                    print('timeout!')
                elif response == b"ACK":
                    self.data.pop(0)
                    nonce_counter += 1
                elif response == b"ACK/RST":
                    print('All data sended')
                    break
                else:
                    print('bad data')

    def sender(self):
        _thread.start_new_thread(self.handler, (self, ))

    def buffer(self, data):
        """Append data for send"""
        self.data.append(data)

    def send(self,data):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((self.IP, self.PORT))
        sock.send(data)
        sock.settimeout(5)
        try:
            response = sock.recv(1024)
        except socket.timeout:
            response = "timeout"
        sock.close()
        return response

    @staticmethod
    def __cipher_RC4():
        key = b'Very long and confidential key'
        nonce = Random.new().read(16)
        tempkey = SHA.new(key + nonce).digest()
        cipher = ARC4.new(tempkey)
        return cipher, nonce

    @staticmethod
    def __cipher_AES():
        key = b'Very long and co'
        nonce = number.getRandomInteger(128)
        ctr = Counter.new(128, initial_value=nonce)
        cipher = AES.new(key, AES.MODE_CTR, counter=ctr)

        return cipher, bytes(str(nonce), 'utf-8')


class Simulator(object):
    """
    Generate a simulated samples vector
    """

    def __init__(self, image, mode=''):
        self.mode = mode
        self.image = image
        self.counter = 0
        self.masks = []
        m = Masks()
        np.random.seed(1)
        if mode == "random":
            self.masks = m.generate_random(
                self.image.shape[0]*self.image.shape[0], self.image.shape[0])
        elif mode == "hadamard":
            self.masks = m.generate_hadamard(
                self.image.shape[0]*self.image.shape[0], self.image.shape[0])

    def get_sample(self):
        try:
            intensity = np.sum(self.masks[self.counter] * self.image)
        except:
            return EOFError
        self.counter += 1
        return intensity


class Image(object):
    def __init__(self):
        """
        Constructor
        """

    def return_image(self, size=64, path=''):
        """
        Return image
        """
        if path == '':
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
    Generate a vector of Masks
    """

    def generate_hadamard(self, number, size):
        """Generate a n hadamard matrix vector"""
        np.random.seed(1)
        ids = list(range(0, size*size))
        np.random.shuffle(ids)
        matrix_vector = []
        for i in range(0, number):
            hadamard_matrix = self.__hadamard(ids[i], size)
            matrix_vector.append(hadamard_matrix)
        np.random.seed(1)
        np.random.shuffle(matrix_vector)
        return matrix_vector

    def generate_random(self, number, size):
        """Generate a random matrix vector"""
        matrix_vector = []
        np.random.seed(1)
        for _ in range(0, number):
            random_matrix = (np.random.rand(size, size) < 0.5) * 1
            matrix_vector.append(random_matrix)
        return matrix_vector

    def __hadamard(self, id_num, size):
        """
        Private Method
        Return a hadamard matrix
        """
        order = int(math.log(size, 2))
        matrix_code = self.__base_convert(id_num, 4)
        padding = np.zeros(
            int(math.log(size**2, 4)) - len(matrix_code), dtype="int")
        vector = np.concatenate((padding, matrix_code))
        vector = vector[::-1]
        v_m = [[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]]
        h_m = np.array([[1]])

        for i in range(0, order):
            h_m1 = np.concatenate((v_m[vector[i]][0] * h_m,
                                   v_m[vector[i]][2] * h_m))
            h_m2 = np.concatenate((v_m[vector[i]][1] * h_m,
                                   v_m[vector[i]][3] * h_m))
            h_m = np.concatenate((h_m1, h_m2), 1)

        return h_m

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


def imshow(image):
    """
    Print image
    """
    plt.figure()
    plt.gray()
    plt.imshow(image)
    plt.show()
