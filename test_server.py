import server
import client

s = server.server()
s.server()
c = client.client()
c.sender()
for i in range(0,110):
    c.buffer(str(i))
while True:
    a = 2
