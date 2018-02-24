"""
Main
"""

import math
import socket
import _thread
import numpy as np
import scipy.misc
import matplotlib.pyplot as plt
import scipy.fftpack as spfft
import cvxpy as cvx
"""
from pylbfgs import owlqn
from Crypto.Cipher import ARC4
from Crypto.Hash import SHA
from Crypto import Random
<<<<<<< HEAD
from Crypto.Cipher import AES
from Crypto.Util import Counter
from Crypto.Util import number
"""

=======
>>>>>>> refs/remotes/origin/master

class Client(object):
    def __init__(self,cipher_mode='AES'):
        self.data = []
        self.cipher_mode=cipher_mode
    def handler(self, _):
        nonce_counter = 1001
        while True:
            if nonce_counter > 1000:
                if self.cipher_mode=='AES':
                    cipher, nonce = self.__cipher_AES()
                elif self.cipher_mode=='RC4':
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
        ctr = Counter.new(128,initial_value=nonce)
        cipher = AES.new(key, AES.MODE_CTR, counter=ctr)
    
        return cipher, bytes(str(nonce),'utf-8')

class Server(object):
    def __init__(self, sock=None, cipher_mode='AES'):
        self.data = []
        if sock is None:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        else:
            self.sock = sock
        self.__nonce = ""
        self.__nonce_recv = "n"
        self.cipher_mode=cipher_mode

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
                    if self.cipher_mode=='AES':
                        cip = self.__cipher_AES(self.__nonce_recv)
                    elif self.cipher_mode=='RC4':
                        cip = self.__cipher_RC4(self.__nonce_recv)
                    self.__nonce = self.__nonce_recv
                decrypted_data = cip.decrypt(recibido[5:])
                print(decrypted_data)
                if decrypted_data == b"END":
                    print("All data recived")
                    sc.send(b"ACK/RST")
                    sc.close()
                    break
                self.data.append(decrypted_data)
                print("data recived: " + str(decrypted_data, 'utf-8'))
                sc.send(b"ACK")

        return self.get_data()

    def get_data(self):
        return self.data

    @staticmethod
    def __cipher_AES(nonce):
        key = b'Very long and co'
        ctr = Counter.new(128,initial_value=int(nonce))
        cipher = AES.new(key, AES.MODE_CTR, counter=ctr)
        return cipher
    @staticmethod
    def __cipher_RC4(nonce):
        key = b'Very long and confidential key'
        tempkey = SHA.new(key + nonce).digest()
        cipher = ARC4.new(tempkey)
        return cipher

class Simulator(object):
    """
    Generate a simulated samples vector
    """

    def __init__(self, image,mode=''):
        self.mode = mode
        self.image = image
<<<<<<< HEAD
        self.counter = 0
        self.masks = []
        m = Masks()
        np.random.seed(1)
        if mode == "random":
            self.masks = m.generate_random(self.image.shape[0]*self.image.shape[0], self.image.shape[0])
        elif mode == "hadamard":
            self.masks = m.generate_hadamard(self.image.shape[0]*self.image.shape[0], self.image.shape[0])
    def get_sample(self):
        try:
            intensity = np.sum(self.masks[self.counter] * self.image)
        except:
            return EOFError
        self.counter+=1
        return intensity


class Single(object):
    def reconstruction(self, samples, image_size, method='', mask=''):
        m = Masks()
      
=======

    def random_samples(self):
        return self.__auxiliar_function(1)
    
    def hadamard_samples(self):
        return self.__auxiliar_function(2)

    def __auxiliar_function(self, t):
        m = Masks()
        if t == 1:
            masks = m.generate_random(self.num_samples, self.image.shape)
        elif t == 2:
            masks = m.generate_hadamard(self.num_samples, self.image.shape)
        samples = []
        for i in masks:
            masked = i * self.image
            samples.append(np.sum(masked))
        return samples


class One(object):
    def reconstruction(self, samples, image_size, method=''):
        m = Masks()
>>>>>>> refs/remotes/origin/master
        self.image_size = image_size
        self.samples = samples
        if mask == 'hadamard':
                masks = m.generate_hadamard(len(samples), image_size)
        else:
            random_masks = m.generate_random(len(samples), image_size)
            masks= []
            for i in random_masks:
                masks.append(i.T.flatten())
        if method == 'direct_inverse':
            matrix_vector = []
            np.random.seed(1)
            for _ in range(0, len(samples)):
                random_matrix = (np.random.rand(self.image_size, self.image_size) < 0.5) * np.float32(1)
                matrix_vector.append(random_matrix.flatten())
            Tmatrix_vector=np.linalg.pinv(np.matrix(matrix_vector))
            res = np.reshape(np.matmul(Tmatrix_vector,np.matrix(samples).T), (64, 64))
        if method == 'hadamard':
<<<<<<< HEAD
            if mask == '' or mask == 'hadamard':
                masks = m.generate_hadamard(len(samples), image_size)
            else:
                masks = random_masks
           
            res = np.zeros([image_size, image_size])
=======
            hadamard_masks = m.generate_hadamard(len(samples), image_size)
            res = np.zeros([image_size[0], image_size[1]])
>>>>>>> refs/remotes/origin/master
            for i in range(0, len(samples)):
                res += masks[i] * samples[i]
        if method == 'fourier':
            random_masks = m.generate_random(len(samples), image_size)
            if mask == 'hadamard' :
                 random_masks = m.generate_hadamard(len(samples), image_size)
                 random_masks = (np.asarray(random_masks)>0)*1.0
            random_masks_formated = []
            for i in random_masks:
                random_masks_formated.append(i.T.flatten())
            A = np.kron(
                spfft.idct(np.identity(image_size[0]), norm='ortho', axis=0),
                spfft.idct(np.identity(image_size[1]), norm='ortho', axis=0))
            A = np.dot(random_masks_formated, A)
            vx = cvx.Variable(image_size[0] * image_size[1])
            objective = cvx.Minimize(cvx.norm(vx, 1))
            constraints = [A * vx == samples]
            prob = cvx.Problem(objective, constraints)
<<<<<<< HEAD
            result = prob.solve(verbose=False)
            Xat2 = np.array(vx.value).squeeze()
            Xat = Xat2.reshape(image_size, image_size).T
            res = self.__idct2(Xat)
=======
            prob.solve(verbose=True)
            result2 = np.array(vx.value).squeeze()
            result = result2.reshape(image_size[0], image_size[1]).T
            res = self.__idct2(result)
>>>>>>> refs/remotes/origin/master
        if method == 'fourier_optim':
            random_masks = m.generate_random(len(samples), image_size)
            if mask == 'hadamard' :
                 random_masks = m.generate_hadamard(len(samples), image_size)
                 random_masks = (np.asarray(random_masks)>0)*1.0
            random_masks_formated = []
            for i in random_masks:
                random_masks_formated.append(i.T.flatten())
            self.random_masks = random_masks_formated
<<<<<<< HEAD
            Xat2 = owlqn(image_size * image_size, self.__evaluate,
                         self.__progress, 500)
            Xat = Xat2.reshape(image_size, image_size).T  # stack columns
            res = self.__idct2(Xat)
=======
            tresult2 = owlqn(image_size[0] * image_size[1], self.__evaluate,
                             self.__progress, 10000)
            tresult = tresult2.reshape(image_size[0], image_size[1]).T
            result = self.__idct2(tresult)
            res = result.astype('uint8').T
>>>>>>> refs/remotes/origin/master
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
        x2 = x.reshape((self.image_size[0], self.image_size[1])).T

        # Ax is just the inverse 2D dct of x2
        Ax2 = self.__idct2(x2)

        Ax = np.dot(self.random_masks, Ax2.flatten())

        # calculate the residual Ax-b and its 2-norm squared

        Axb = Ax - self.samples

        fx = np.sum(np.power(Axb, 2))

        Axb2 = np.zeros(x2.shape, dtype="float64")
        for a in range(0, len(self.random_masks)):
            Axb2 += self.random_masks[a].reshape(x2.shape) * Axb[a]

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
        #print(gnorm)
       
        if gnorm < 0.01:
            a = 1
        else:
            print(gnorm)
            a = 0
        return 0
        

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
<<<<<<< HEAD
        """Generate a n hadamard matrix vector"""
        np.random.seed(1)
        ids =list(range(0,size*size))
        np.random.shuffle(ids)
        matrix_vector = []
        for i in range(0, number):
            hadamard_matrix = self.__hadamard(ids[i], size)
=======
        """Generate a hadamard matrix vector"""
        if size[0]!=size[1]:
            exit()
        matrix_vector = []
        for i in range(0, number):
            hadamard_matrix = self.__hadamard(i, size[0])
>>>>>>> refs/remotes/origin/master
            matrix_vector.append(hadamard_matrix)
        np.random.seed(1)
        np.random.shuffle(matrix_vector)
        return matrix_vector

    def generate_random(self, number, size):
        """Generate a random matrix vector"""
        matrix_vector = []
        np.random.seed(1)
        for _ in range(0, number):
            random_matrix = (np.random.rand(size[0], size[1]) < 0.5) * 1
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