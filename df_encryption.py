import scipy.misc
import numpy as np
import matplotlib.pyplot as plt
import cmath
TEST_IMAGE = scipy.misc.face()
TEST_IMAGE = TEST_IMAGE[:, :, 1]
TEST_IMAGE = scipy.misc.imresize(TEST_IMAGE, [256, 256])
x,y = TEST_IMAGE.shape

print cmath.exp(2j)
"""
im = 0+255j
RE_RFM1 = np.random.rand(x,y)*255
IM_RFM1 = (np.random.rand(x,y))*im
RFM1 = IM_RFM1+RE_RFM1
RE_RFM2 = np.random.rand(x,y)*255
IM_RFM2 = (np.random.rand(x,y))*im
RFM2 = IM_RFM2+RE_RFM2
"""
RFM1 = np.ones([x,y],dtype="complex")
RFM2 = np.ones([x,y],dtype="complex")
for i in range(0,x):
    for j in range(0,y):
        RFM1[i,j]=cmath.exp(2*3.14*1j*np.random.rand(1))
        RFM2[i,j]=cmath.exp(2*3.14*1j*np.random.rand(1))

TEST_IMAGE_RFM1 = TEST_IMAGE*RFM1
F_TEST_IMAGE = np.fft.fft2(TEST_IMAGE_RFM1)
F_TEST_IMAGE_RFM2 = F_TEST_IMAGE*RFM2
EN_TEST_IMAGE = np.fft.ifft2(F_TEST_IMAGE_RFM2)

F_EN_TEST_IMAGE = np.fft.fft2(EN_TEST_IMAGE)
F_EN_TEST_IMAGE_RFM2 = F_EN_TEST_IMAGE/RFM2
RE_TEST_IMAGE = np.fft.ifft2(F_EN_TEST_IMAGE_RFM2)




plt.figure()
plt.subplot(1,3,1)
plt.imshow(TEST_IMAGE,"gray")
plt.subplot(1,3,2)
plt.imshow(abs(EN_TEST_IMAGE),"gray")
plt.subplot(1,3,3)
plt.imshow(abs(RE_TEST_IMAGE),"gray")
plt.show()