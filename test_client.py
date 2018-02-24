import main
import time

c = main.Client()
c.sender()
print("Comunication initiated")
for i in range(0,5):
    print(bytes(str(i),'utf-8'))
    c.buffer(bytes(str(i),'utf-8'))
time.sleep(1)
print(b'END')
c.buffer(b'END')
time.sleep(1)