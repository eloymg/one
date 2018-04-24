import main_raspy as main
import time

c = main.Client()
c.sender()
print("Comunication initiated")
i = main.Image()
im = i.return_image(size=32)
s = main.Simulator(im,mode="hadamard")
for i in range(0,32*32*1):
    sam = bytes(str(s.get_sample()),'utf-8')
    c.buffer(sam)
time.sleep(1)
print(b'END')
c.buffer(b'END')
time.sleep(1)