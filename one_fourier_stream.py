
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as spopt
import scipy.fftpack as spfft
import scipy.ndimage as spimg
import scipy.misc
import cvxpy as cvx

from Crypto.Cipher import ARC4
from Crypto.Hash import SHA
from Crypto import Random

def dct2(x):
    return spfft.dct(spfft.dct(x.T, norm='ortho', axis=0).T, norm='ortho', axis=0)

def idct2(x):
    return spfft.idct(spfft.idct(x.T, norm='ortho', axis=0).T, norm='ortho', axis=0)

# read original image and downsize for speed

TEST_IMAGE = scipy.misc.face()
#TEST_IMAGE = plt.imread("/Users/eloymorenogarcia/Desktop/R.jpg")
TEST_IMAGE = TEST_IMAGE[:, :, 1]
X = scipy.misc.imresize(TEST_IMAGE, [32, 32])
ny,nx = X.shape
ny = int(ny)
nx = int(nx)

# create dct matrix operator using kron (memory errors for large ny*nx)
A = np.kron(
    spfft.idct(np.identity(nx), norm='ortho', axis=0),
    spfft.idct(np.identity(ny), norm='ortho', axis=0)
    )
mask_vec =[]
intensity_vec =[]
for i in range(0,500):
    mask = (np.random.rand(ny,nx)<0.5)*1
    masked = mask*X
    intensity = np.sum(masked)
    mask_vec.append(mask.T.flatten())
    intensity_vec.append(intensity)
mask_vec = np.expand_dims(mask_vec, axis=1)
print "first step"
A = np.dot(mask_vec,A) # same as phi times kron


#Intensity_vec--> encrypted
key = 'Very long and confidential key'
nonce = Random.new().read(16)
tempkey = SHA.new(key+nonce).digest()
cipher1 = ARC4.new(tempkey)
cipher_msg = []
cipher_msg.append(nonce)
for i in intensity_vec:
    cipher_msg.append(cipher1.encrypt(str(i)))
#--------------------------
#Intensity_vec--> decrypted
intensity_vec_decrypted = []
key2 = 'Very long and confidential key'
nonce2 = cipher_msg.pop(0)
tempkey2 = SHA.new(key2+nonce2).digest()
cipher2 = ARC4.new(tempkey2)
for i in cipher_msg:
    intensity_vec_decrypted.append(cipher2.decrypt(i))
#-----------------------------------

# do L1 optimization
vx = cvx.Variable(nx * ny)
objective = cvx.Minimize(cvx.norm(vx, 1))
constraints = [A*vx == intensity_vec_decrypted]
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