import socket
import thread
import time

class client:
    def __init__(self, sock=None):
        self.data = []
    def handler(self,a):
        while True:
            time.sleep(3)
            if len(self.data)>0:
                self.sock = socket.socket()
                self.sock.connect(("127.0.0.1", 9999))
                self.sock.send(self.data.pop())
                self.sock.close()
    def sender(self):
       thread.start_new_thread(self.handler,(self,))
    def send(self,d):
        self.data.append(d)
        

