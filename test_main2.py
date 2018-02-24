import main
import matplotlib.pyplot as plt
from skimage.measure import compare_ssim as ssim
from skimage.exposure import rescale_intensity as rescale
import numpy as np
import time
tic = time.clock()
toc = time.clock()
toc - tic
"""
c = main.Client()
c.sender()
"""
def imshow(im):
    plt.figure()
    plt.gray()
    plt.imshow(im)
    plt.show()

i = main.Image()
per = np.arange(0.1, 1.1, 0.1)
for j in per:
    i = main.Image()
    im = i.return_image(size=64)
    o = main.Single()
  
    tic = time.clock()
    s = main.Simulator(im,mode="random")
    samples = []
    for i in range(0,int((64*64)*j)):
        samples.append(s.get_sample())
        """
        c.buffer(bytes(str(s.get_sample()),'utf-8'))
        """
    res1 = o.reconstruction(samples,im.shape[0],method='direct_inverse')
    toc = time.clock()
    t1 = toc - tic
    s1 = ssim(rescale(res1,out_range=(0,255)),im)
    print(str(j)+" "+str(s1)+" "+str(t1))
    
    
    
    
    
