import main
import matplotlib.pyplot as plt
from skimage.measure import compare_ssim as ssim
from skimage.exposure import rescale_intensity as rescale
import numpy as np

s = main.Server()
s.handler()
i = main.Image()
im = i.return_image(size=32)
hadamard_samples = s.get_data()
o = main.Single()
res1 = o.reconstruction(hadamard_samples,32,method='hadamard',mask='hadamard')
s1 = ssim(rescale(res1,out_range=(0,255)).astype("uint8"),im.astype("uint8"))
print(s1)
plt.imshow(res1)