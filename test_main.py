import main
import matplotlib.pyplot as plt
from skimage.measure import compare_ssim as ssim

def imshow(im):
    plt.figure()
    plt.gray()
    plt.imshow(im)
    plt.show()

s = main.Server()
d = s.handler()
print(d)
exit()
i = main.Image()
im = i.return_image()
s = main.Simulator(500,im)
o = main.One()
res = o.reconstruction(s.hadamard_samples(),im.shape[0],method='fourier_optim')
imshow(res)
res = o.reconstruction(s.hadamard_samples(),im.shape[0],method='hadamard')
imshow(res)
res = o.reconstruction(s.random_samples(),im.shape[0],method='fourier')
imshow(res)
