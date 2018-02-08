import main
import matplotlib.pyplot as plt
from skimage.measure import compare_ssim as ssim

def imshow(im):
    plt.figure()
    plt.gray()
    plt.imshow(im)
    plt.show()
i = main.Image()

im = i.return_image(size=64)
s = main.Simulator(im,mode="random")
o = main.One()
hadamard_samples = []
for i in range(0,int((64*64)*1)):
    hadamard_samples.append(s.get_sample())

res1 = o.reconstruction(hadamard_samples,im.shape[0],method='hadamard',mask='random')
imshow(res1)
imshow(im)
print(ssim(res1,im))