
import numpy as np
import matplotlib.pyplot as plt

import scipy.misc
import cvxpy as cvx
import pickle
import scipy.fftpack as spfft
from skimage.measure import compare_ssim as ssim

def idct2(x):
    return spfft.idct(spfft.idct(x.T, norm='ortho', axis=0).T, norm='ortho', axis=0)

TEST_IMAGE = scipy.misc.face()
#TEST_IMAGE = plt.imread("/Users/eloymorenogarcia/Desktop/R.jpg")
TEST_IMAGE = TEST_IMAGE[:, :, 1]
X = scipy.misc.imresize(TEST_IMAGE, [64, 64])
ny,nx = X.shape
ny = int(ny)
nx = int(nx)

file = open('dict_hadamard.txt', 'r')
A = pickle.load(file)

file = open('masks_hadamard.txt', 'r')
mask_vec = pickle.load(file)

ssim_vec = []
M_vec = []
for M in range(96,4096+800,800):

   # M = 500

    intensity_vec = []

    for i in range(0,M):
        mask = mask_vec[i]
        masked = mask*X
        intensity = np.sum(masked)
        intensity_vec.append(intensity)

    # do L1 optimization
    AX = A[0:M,:]
    print AX.shape
    vx = cvx.Variable(nx * ny)
    objective = cvx.Minimize(cvx.norm(vx, 1))
    constraints = [AX*vx == intensity_vec]
    prob = cvx.Problem(objective, constraints)
    result = prob.solve(verbose=True)
    Xat2 = np.array(vx.value).squeeze()

    # reconstruct signal
    Xat = Xat2.reshape(nx, ny).T # stack columns
    Xa = idct2(Xat)
    Xa = Xa.astype("float64")
    X = X.astype("float64")
    print ssim(Xa,X)
    ssim_vec.append(ssim(Xa,X))
    M_vec.append(M)
    
plt.figure()
plt.plot(M_vec,ssim_vec)
plt.show()