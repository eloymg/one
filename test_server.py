import server
import client

s = server.mysocket()
s.server()

c = client.mysocket()
c.sender()
c.send("adata:ad2asd")
c.send("adata:ada3sd")
c.send("adata:ada5sd")
print c.data

while True:
    a = 2