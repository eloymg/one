
import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack as spfft
import scipy.ndimage as spimg
import pickle

def dct2(x):
    return spfft.dct(spfft.dct(x.T, norm='ortho', axis=0).T, norm='ortho', axis=0)

def idct2(x):
    return spfft.idct(spfft.idct(x.T, norm='ortho', axis=0).T, norm='ortho', axis=0)

nx=64
ny=64

A = np.kron(
    spfft.idct(np.identity(nx), norm='ortho', axis=0),
    spfft.idct(np.identity(ny), norm='ortho', axis=0)
    )
mask_vec =[]
mask_vec_raw =[]
for i in range(0,nx*ny):
    mask = (np.random.rand(ny,nx)<0.5)*1
    mask_vec_raw.append(mask)
    mask_vec.append(mask.T.flatten())
mask_vec = np.expand_dims(mask_vec, axis=1)
A = np.dot(mask_vec,A)

file = open('dict.txt', 'w')
pickle.dump(A, file)
file.close()
file = open('masks.txt', 'w')
pickle.dump(mask_vec_raw, file)
file.close()