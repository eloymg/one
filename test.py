import cv2
import matplotlib.pyplot as plt
import scipy.misc
import time

camera = cv2.VideoCapture(0)
 
# Captures a single image from the camera and returns it in PIL format
def get_image():
 # read is the easiest way to get a full image out of a VideoCapture object.
 retval,  im = camera.read()
 return im

im_vec=[]

tic = time.clock()

for i in range(0, 1000):
    im = scipy.misc.imresize(get_image(), [128, 128])
    im_vec.append(im)

toc = time.clock()

print (toc-tic)