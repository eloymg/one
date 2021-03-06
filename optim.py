
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.fftpack as spfft
import scipy.misc
from skimage.measure import compare_ssim as ssim
import time

from pylbfgs import owlqn


def dct2(x):
    return spfft.dct(
        spfft.dct(x.T, norm='ortho', axis=0).T, norm='ortho', axis=0)


def idct2(x):
    return spfft.idct(
        spfft.idct(x.T, norm='ortho', axis=0).T, norm='ortho', axis=0)


def progress(x, g, fx, xnorm, gnorm, step, k, ls):
    # Print variables to screen or file or whatever. Return zero to
    # continue algorithm; non-zero will halt execution.

    if gnorm < 5:
        a = 1
    else:
        a = 0
    return a


def evaluate(x, g, step):
    """An in-memory evaluation callback."""

    # we want to return two things:
    # (1) the norm squared of the residuals, sum((Ax-b).^2), and
    # (2) the gradient 2*A'(Ax-b)

    # expand x columns-first
    x2 = x.reshape((nx, ny)).T

    # Ax is just the inverse 2D dct of x2
    Ax2 = idct2(x2)
    im.set_data(Ax2)
    fig.canvas.draw()
    print(ssim(Xorig, Ax2))

    # stack columns and extract samples

    ############OPTION1
    """
    Ax = Ax2.T.flat[ri].reshape(b.shape)
    """
    ####OPTIONA2
    Ax = np.dot(mask_vec, Ax2.T.flatten())
    ######
    # calculate the residual Ax-b and its 2-norm squared

    ############OPTION1
    """
    Axb = Ax - b
    """
    ####OPTIONA2

    Axb = Ax - intensity_vec
    #####
    fx = np.sum(np.power(Axb, 2))
    # project residual vector (k x 1) onto blank image (ny x nx)

    ############OPTION1
    """
    Axb2 = np.zeros(x2.shape)
    Axb2.T.flat[ri] = Axb
    """
    ####OPTIONA2

    Axb2 = np.zeros(x2.shape, dtype="float64")
    for a in range(0, len(mask_vec)):
        Axb2 += mask_vec[a].reshape(x2.shape).T * Axb[a]
    """
    Axb2 = np.dot(Axb,mask_vec).reshape(x2.shape).T
    """

    # A'(Ax-b) is just the 2D dct of Axb2
    AtAxb2 = 2 * dct2(Axb2)
    AtAxb = AtAxb2.T.reshape(x.shape)  # stack columns
    # copy over the gradient vector

    np.copyto(g, AtAxb)

    return fx


# read original image
TEST_IMAGE = scipy.misc.face()
TEST_IMAGE = TEST_IMAGE[:, :, 0]
Xorig = scipy.misc.imresize(TEST_IMAGE, [128, 128])
Xorig = Xorig.astype("float64")
ny, nx = Xorig.shape

mask_vec = []
intensity_vec = []

for i in range(0, 1000):
    mask = (np.random.rand(nx, ny) < 0.5) * 1
    masked = mask * Xorig
    intensity = np.sum(masked)
    mask_vec.append(mask.T.flatten())
    intensity_vec.append(intensity)
#mask_vec = np.expand_dims(mask_vec, axis=1)
mask_vec = np.asarray(mask_vec)
intensity_vec = np.asarray(intensity_vec, dtype="float64")

# perform the L1 minimization in memory
tic = time.clock()
plt.gray()
plt.ion()
fig = plt.figure()
im = plt.imshow(Xorig)
plt.show()
Xat2 = owlqn(nx * ny, evaluate, progress, 10000)
tac = time.clock()

print(tac - tic)

# transform the output back into the spatial domain
Xat = Xat2.reshape(nx, ny).T  # stack columns
Xa = idct2(Xat)
Z = Xa.astype('uint8')

fig = plt.figure()
fig.add_subplot(1, 2, 1)
plt.gray()
plt.imshow(Xorig)
fig.add_subplot(1, 2, 2)
plt.imshow(Z)
plt.show()