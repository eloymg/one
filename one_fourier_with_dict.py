
import numpy as np
import matplotlib.pyplot as plt

import scipy.misc
import cvxpy as cvx
import pickle
import scipy.fftpack as spfft

def idct2(x):
    return spfft.idct(spfft.idct(x.T, norm='ortho', axis=0).T, norm='ortho', axis=0)

TEST_IMAGE = scipy.misc.face()
#TEST_IMAGE = plt.imread("/Users/eloymorenogarcia/Desktop/R.jpg")
TEST_IMAGE = TEST_IMAGE[:, :, 1]
X = scipy.misc.imresize(TEST_IMAGE, [32, 32])
ny,nx = X.shape
ny = int(ny)
nx = int(nx)

file = open('dict.txt', 'r')
A = pickle.load(file)

file = open('masks.txt', 'r')
mask_vec = pickle.load(file)

M = 500

intensity_vec = []

for i in range(0,M):
    masked = mask_vec[i]*X
    intensity = np.sum(masked)
    intensity_vec.append(intensity)

# do L1 optimization
A = A[0:M,:]
vx = cvx.Variable(nx * ny)
objective = cvx.Minimize(cvx.norm(vx, 1))
constraints = [A*vx == intensity_vec]
prob = cvx.Problem(objective, constraints)
result = prob.solve(verbose=True)
Xat2 = np.array(vx.value).squeeze()

# reconstruct signal
Xat = Xat2.reshape(nx, ny).T # stack columns
Xa = idct2(Xat)


plt.figure()
plt.subplot(1,2,1)
plt.imshow(X,"gray")
plt.subplot(1,2,2)
plt.imshow(Xa,"gray")
plt.show()