import main
import time

c = main.Client()
c.sender()
print("Comunication initiated")
for i in range(0,5):
    time.sleep(2)
    c.buffer(str(i))
c.send('END')
