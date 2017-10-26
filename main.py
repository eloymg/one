"""
Main
"""

import math
import socket
import time
import _thread
import numpy as np
import scipy.misc
import matplotlib.pyplot as plt
import scipy.fftpack as spfft
import cvxpy as cvx
from pylbfgs import owlqn
from Crypto.Cipher import ARC4
from Crypto.Hash import SHA
from Crypto import Random

"""
for i in intensity_vec:
    cipher_msg.append(cipher1.encrypt(str(i)))
"""


class Client(object):
    def __init__(self):
        self.data = []

    def handler(self, _):
        nonce_counter = 1001
        while True:
            if nonce_counter > 1000:
                cipher, nonce = self.__cipher()
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

    @staticmethod
    def send(data):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect(("127.0.0.1", 9999))
        sock.send(data)
        sock.settimeout(5)
        try:
            response = sock.recv(1024)
        except socket.timeout:
            response = "timeout"
        sock.close()
        return response

    @staticmethod
    def __cipher():
        key = b'Very long and confidential key'
        nonce = Random.new().read(16)
        tempkey = SHA.new(key + nonce).digest()
        cipher = ARC4.new(tempkey)
        return cipher, nonce


class Server(object):
    def __init__(self, sock=None):
        self.data = []
        if sock is None:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        else:
            self.sock = sock
        self.__nonce = ""
        self.__nonce_recv = "n"

    def handler(self):
        self.sock.bind(("", 9999))
        self.sock.listen(1)
        sc, _ = self.sock.accept()
        print("Comunication initiated")
        while True:
            recibido = sc.recv(100)
            if recibido == b"":
                sc.close()
                sc, _ = self.sock.accept()
            if recibido[:6] == b"nonce:":
                self.__nonce_recv = recibido[6:]
                print("Nonce recived: " + str(self.__nonce_recv))
                sc.send(b"ACK")
            if recibido[:5] == b"data:":
                if self.__nonce_recv != self.__nonce:
                    cip = self.__cipher(self.__nonce_recv)
                    self.__nonce = self.__nonce_recv
                decrypted_data = cip.decrypt(recibido[5:])
                print(decrypted_data)
                if decrypted_data == "END":
                    print("All data recived")
                    sc.send(b"ACK/RST")
                    sc.close()
                    break
                self.data.append(decrypted_data)
                print("data recived: " + str(decrypted_data,'utf-8'))
                sc.send(b"ACK")
            
        return self.get_data()

    def get_data(self):
        return self.data

    @staticmethod
    def __cipher(nonce):
        key = b'Very long and confidential key'
        tempkey = SHA.new(key + nonce).digest()
        cipher = ARC4.new(tempkey)
        return cipher


class Simulator(object):
    """
    Generate a simulated samples vector
    """

    def __init__(self, num_samples, image):
        self.num_samples = num_samples
        self.image = image

    def hadamard_samples(self):
        return self.__auxiliar_function("h")

    def random_samples(self):
        return self.__auxiliar_function("r")

    def __auxiliar_function(self, t):
        m = Masks()
        np.random.seed(1)
        if t == "r":
            masks = m.generate_random(self.num_samples, self.image.shape[0])
        elif t == "h":
            masks = m.generate_hadamard(self.num_samples, self.image.shape[0])
            np.random.shuffle(masks)
        samples = []
        for i in masks:
            masked = i * self.image
            samples.append(np.sum(masked))
        return samples


class One(object):
    def reconstruction(self, samples, image_size, method=''):
        m = Masks()
        np.random.seed(1)
        self.image_size = image_size
        self.samples = samples
        if method == 'hadamard':
            hadamard_masks = m.generate_hadamard(len(samples), image_size)
            np.random.shuffle(hadamard_masks)
            res = np.zeros([image_size, image_size])
            for i in range(0, len(samples)):
                res += hadamard_masks[i] * samples[i]
        if method == 'fourier':
            random_masks = m.generate_random(len(samples), image_size)
            random_masks_formated = []
            for i in random_masks:
                random_masks_formated.append(i.T.flatten())
            A = np.kron(
                spfft.idct(np.identity(image_size), norm='ortho', axis=0),
                spfft.idct(np.identity(image_size), norm='ortho', axis=0))
            A = np.dot(random_masks_formated, A)
            vx = cvx.Variable(image_size * image_size)
            objective = cvx.Minimize(cvx.norm(vx, 1))
            constraints = [A * vx == samples]
            prob = cvx.Problem(objective, constraints)
            result = prob.solve(verbose=True)
            Xat2 = np.array(vx.value).squeeze()
            Xat = Xat2.reshape(image_size, image_size).T
            res = self.__idct2(Xat)
        if method == 'fourier_optim':
            random_masks = m.generate_random(len(samples), image_size)
            random_masks_formated = []
            for i in random_masks:
                random_masks_formated.append(i.T.flatten())
            self.random_masks = random_masks_formated
            Xat2 = owlqn(image_size * image_size, self.__evaluate,
                         self.__progress, 10000)
            Xat = Xat2.reshape(image_size, image_size).T  # stack columns
            Xa = self.__idct2(Xat)
            res = Xa.astype('uint8')
        return res

    @staticmethod
    def __idct2(x):
        return spfft.idct(
            spfft.idct(x.T, norm='ortho', axis=0).T, norm='ortho', axis=0)

    @staticmethod
    def __dct2(x):
        return spfft.dct(
            spfft.dct(x.T, norm='ortho', axis=0).T, norm='ortho', axis=0)

    def __evaluate(self, x, g, step):
        """An in-memory evaluation callback."""

        # we want to return two things:
        # (1) the norm squared of the residuals, sum((Ax-b).^2), and
        # (2) the gradient 2*A'(Ax-b)

        # expand x columns-first
        x2 = x.reshape((self.image_size, self.image_size)).T

        # Ax is just the inverse 2D dct of x2
        Ax2 = self.__idct2(x2)

        Ax = np.dot(self.random_masks, Ax2.T.flatten())

        # calculate the residual Ax-b and its 2-norm squared

        Axb = Ax - self.samples

        fx = np.sum(np.power(Axb, 2))

        Axb2 = np.zeros(x2.shape, dtype="float64")
        for a in range(0, len(self.random_masks)):
            Axb2 += self.random_masks[a].reshape(x2.shape).T * Axb[a]

        # A'(Ax-b) is just the 2D dct of Axb2
        AtAxb2 = 2 * self.__dct2(Axb2)
        AtAxb = AtAxb2.T.reshape(x.shape)  # stack columns
        # copy over the gradient vector

        np.copyto(g, AtAxb)

        return fx

    @staticmethod
    def __progress(x, g, fx, xnorm, gnorm, step, k, ls):
        # Print variables to screen or file or whatever. Return zero to
        # continue algorithm; non-zero will halt execution.
        if gnorm < 5:
            a = 1
        else:
            a = 0
        return a


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
    todo
    """

    def generate_hadamard(self, number, size):
        """Generate a n hadamard matrix vector"""
        matrix_vector = []
        for i in range(0, number):
            hadamard_matrix = self.__hadamard(i, size)
            matrix_vector.append(hadamard_matrix)
        return matrix_vector

    def generate_random(self, number, size):
        """Generate a n random matrix vector"""
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


def imshow(im):
    plt.figure()
    plt.gray()
    plt.imshow(im)
    plt.show()