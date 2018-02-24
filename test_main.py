import main
import matplotlib.pyplot as plt
from skimage.measure import compare_ssim as ssim
from skimage.exposure import rescale_intensity as rescale
import numpy as np
import time
tic = time.clock()
toc = time.clock()
toc - tic

def imshow(im):
    plt.figure()
    plt.gray()
    plt.imshow(im)
    plt.show()

i = main.Image()
im = i.return_image(size=64).astype("uint8")
per = np.arange(0.1, 1.1, 0.1)
for j in per:
    o = main.Single()
    """
    res = o.reconstruction(s.hadamard_samples(),im.shape[0],method='fourier_optim')
    imshow(res)
    res = o.reconstruction(s.random_samples(),im.shape[0],method='fourier')
    imshow(res)
    """
    tic = time.clock()
    s = main.Simulator(im,mode="hadamard")
    hadamard_samples = []
    for i in range(0,int((64*64)*j)):
        hadamard_samples.append(s.get_sample())
    res1 = o.reconstruction(hadamard_samples,im.shape[0],method='hadamard',mask='hadamard')
    toc = time.clock()
    t1 = toc - tic
    s1 = ssim(rescale(res1,out_range=(0,255)).astype("uint8"),im)
    plt.imsave("H_"+str(j)+".jpg",res1,cmap=plt.cm.gray)
    """
    tic = time.clock()
    s = main.Simulator(im,mode="random")
    random_samples = []
    for i in range(0,int((64*64)*j)):
        random_samples.append(s.get_sample())
    res2 = o.reconstruction(random_samples,im.shape[0],method='fourier',mask="random")
    toc = time.clock()
    t2 = toc - tic
    s2 = ssim(res2.astype("uint8"),im)
    plt.imsave("F_"+str(j)+".jpg",res2,cmap=plt.cm.gray)

    tic = time.clock()
    s = main.Simulator(im,mode="random")
    random_samples = []
    for i in range(0,int((64*64)*j)):
        random_samples.append(s.get_sample())
    res3 = o.reconstruction(random_samples,im.shape[0],method='fourier_optim',mask="random")
    toc = time.clock()
    t3 = toc - tic
    s3 = ssim(res3.astype("uint8"),im)
    plt.imsave("FO_"+str(j)+".jpg",res3,cmap=plt.cm.gray)
    

    print(str(j)+" "+str(s1)+" "+str(s2)+" "+str(s3)+" "+str(t1)+" "+str(t2)+" "+str(t3))
   """
    print(str(j)+" "+str(s1)+" "+str(t1))
    
    
    
