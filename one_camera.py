import math
import scipy.misc
import matplotlib.pyplot as plt
import numpy as np
import cv2

camera = cv2.VideoCapture(0)

def get_image():
 # read is the easiest way to get a full image out of a VideoCapture object.
 retval,  im = camera.read()
 return im

def base_convert(i,  b):
    result = []
    if i == 0:
        return [0]
    while i > 0:
            result.insert(0,  i % b)
            i = i // b
    return result

def hadamard(size, vector):

    # size is 2 power to order
    order = int(math.log(size,  2))
   
    v_m = [[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]]
    h_m = np.array([[1]])

    for i in xrange(0, order):
        h_m1 = np.concatenate((v_m[vector[i]][0]*h_m, v_m[vector[i]][1]*h_m))
        h_m2 = np.concatenate((v_m[vector[i]][2]*h_m, v_m[vector[i]][3]*h_m))
        h_m = np.concatenate((h_m1, h_m2), 1)
    
    return h_m
def generate_hadamard(n):
    matrix_vector = []
    for i in xrange(0, n):
        c = base_convert(i,  4) 
        padding = np.zeros(7-len(c), dtype="int")      
        c = np.concatenate((padding, c))      
        had = hadamard(128, c)
        had = had[:, :]==1
        had = 1*had
       
        matrix_vector.append(had)

    return matrix_vector

hadamard_masks = generate_hadamard(4**7)
np.random.shuffle(hadamard_masks)
num=1

im_vec=[]
imM = scipy.misc.imresize(get_image(), [128, 128])
for i in range(0, 1000):
    im = scipy.misc.imresize(get_image(), [128, 128])
    im_vec.append(im)

res = np.zeros(imM.shape)


for i in range(0, len(hadamard_masks)):
    masked = im_vec[i]*hadamard_masks[i]
    intensity = masked.sum(dtype="float64")
    pixels=m.sum(dtype="float64")
    res += (((intensity/pixels)/4**7)*m)
    num+=1

fig = plt.figure()
a=fig.add_subplot(1,  2,  1)
plt.gray()
plt.imshow(imM)
a=fig.add_subplot(1,  2,  2)
plt.imshow(res)
plt.show()

