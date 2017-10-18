# make sure you've got the following packages installed
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.fftpack as spfft
import scipy.misc

from pylbfgs import owlqn

def dct2(x):
    return spfft.dct(spfft.dct(x.T, norm='ortho', axis=0).T, norm='ortho', axis=0)

def idct2(x):
    return spfft.idct(spfft.idct(x.T, norm='ortho', axis=0).T, norm='ortho', axis=0)

def evaluate(x, g, step):
    """An in-memory evaluation callback."""

    # we want to return two things: 
    # (1) the norm squared of the residuals, sum((Ax-b).^2), and
    # (2) the gradient 2*A'(Ax-b)

    # expand x columns-first
    x2 = x.reshape((nx, ny)).T

    # Ax is just the inverse 2D dct of x2
    Ax2 = idct2(x2)
    
    # stack columns and extract samples

    ############OPTION1
    """
    Ax = Ax2.T.flat[ri].reshape(b.shape)
    """
    ####OPTIONA2
    Ax = np.dot(mask_vec,Ax2.T.flatten())
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
    Axb2 = np.zeros(x2.shape)
    for i in range(0,len(mask_vec)):
        Axb2 += mask_vec[i].reshape((nx,ny)).T*Axb[i]
    #####

    # fill columns-first

    # A'(Ax-b) is just the 2D dct of Axb2
    AtAxb2 = 2 * dct2(Axb2)
    AtAxb = AtAxb2.T.reshape(x.shape) # stack columns

    # copy over the gradient vector
    np.copyto(g, AtAxb)

    return fx



# read original image
TEST_IMAGE = scipy.misc.face()
TEST_IMAGE = TEST_IMAGE[:,:,0]
Xorig = scipy.misc.imresize(TEST_IMAGE, [16, 16])
#Xorig = plt.imread('Escher_Waterfall.jpg')
ny,nx= Xorig.shape

mask_vec =[]
intensity_vec =[]

for i in range(0,100):
    mask = (np.random.rand(nx,ny)<0.5)*1
    masked = mask*Xorig
    intensity = np.sum(masked)
    mask_vec.append(mask.T.flatten())
    intensity_vec.append(intensity)
#mask_vec = np.expand_dims(mask_vec, axis=1)
mask_vec = np.asarray(mask_vec)

# fractions of the scaled image to randomly sample at
sample_size = 0.2

# for each sample size
Z = np.zeros(Xorig.shape, dtype='uint8')
mask = np.zeros(Xorig.shape, dtype='uint8')

   
    # create random sampling index vector
k = round(nx * ny * sample_size)
ri = np.random.choice(nx * ny, k, replace=False) # random sample of indices

# for each color channel
   

# extract channel
X = Xorig.squeeze()

# create images of mask (for visualization)
Xm = 255 * np.ones(X.shape)
Xm.T.flat[ri] = X.T.flat[ri]
mask = Xm

# take random samples of image, store them in a vector b
b = X.T.flat[ri].astype(float)

# perform the L1 minimization in memory
Xat2 = owlqn(nx*ny,evaluate,None, 5)

# transform the output back into the spatial domain
Xat = Xat2.reshape(nx, ny).T # stack columns
Xa = idct2(Xat)
Z = Xa.astype('uint8')


fig = plt.figure()
fig.add_subplot(1, 2, 1)
plt.gray()
plt.imshow(Xorig)
fig.add_subplot(1, 2, 2)
plt.imshow(Z)
plt.show()