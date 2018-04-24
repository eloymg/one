
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as spopt
import scipy.fftpack as spfft
import scipy.ndimage as spimg
import scipy.misc
import cvxpy as cvx

def dct2(x):
    return spfft.dct(spfft.dct(x.T, norm='ortho', axis=0).T, norm='ortho', axis=0)

def idct2(x):
    return spfft.idct(spfft.idct(x.T, norm='ortho', axis=0).T, norm='ortho', axis=0)

# read original image and downsize for speed

TEST_IMAGE = scipy.misc.face()
#TEST_IMAGE = plt.imread("/Users/eloymorenogarcia/Desktop/R.jpg")
TEST_IMAGE = TEST_IMAGE[:, :, 1]
X = scipy.misc.imresize(TEST_IMAGE, [128, 128])
ny,nx = X.shape
ny = int(ny)
nx = int(nx)

# create dct matrix operator using kron (memory errors for large ny*nx)
A = np.kron(
    spfft.idct(np.identity(64), norm='ortho', axis=0),
    spfft.idct(np.identity(64), norm='ortho', axis=0)
    )
mask_vec =[]
intensity_vec =[]

maskS = np.zeros([nx,ny])
maskS[32:64+32, 32:64+32] = 1


intensity_base = np.sum(np.logical_not(maskS)*X)

plt.figure()
plt.imshow(np.logical_not(maskS)*X)
plt.show()

for i in range(0,800):
    mask = (np.random.rand(nx,ny)<0.5)*1*maskS
    mask = np.logical_not(mask)*1
    masked = mask*X
    intensity = np.sum(masked)
    mask_vec.append(mask[32:64+32, 32:64+32].T.flatten())
    print intensity-intensity_base
    intensity_vec.append(intensity-intensity_base)
mask_vec = np.expand_dims(mask_vec, axis=1)
print "first step"
A = np.dot(mask_vec,A) # same as phi times kron

# do L1 optimization
vx = cvx.Variable(64 * 64)
objective = cvx.Minimize(cvx.norm(vx, 1))
constraints = [A*vx == intensity_vec]
prob = cvx.Problem(objective, constraints)
result = prob.solve(verbose=True)
Xat2 = np.array(vx.value).squeeze()

# reconstruct signal
Xat = Xat2.reshape(64, 64).T # stack columns
Xa = idct2(Xat)


plt.figure()
plt.subplot(1,2,1)
plt.imshow(X,"gray")
plt.subplot(1,2,2)
plt.imshow(Xa,"gray")
plt.show()