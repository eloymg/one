import socket
import thread
import time
from Crypto.Cipher import ARC4
from Crypto.Hash import SHA
from Crypto import Random
"""
for i in intensity_vec:
    cipher_msg.append(cipher1.encrypt(str(i)))
"""
class client:
    def __init__(self, sock=None):
        self.data = []
        self.__key = 'Very long and confidential key'
        self.__nonce = Random.new().read(16)
        self.__tempkey = SHA.new(self.__key+self.__nonce).digest()
        self.__cipher = ARC4.new(self.__tempkey)
    def handler(self,a):  
        print "nonce:"+self.__nonce
        self.send("nonce:"+self.__nonce)
        while True:
            if len(self.data)>0:
                print "send!"
                self.sock = socket.socket()
                self.sock.connect(("127.0.0.1", 9999))
                self.sock.send(self.data.pop(0))
                self.sock.close()
    def sender(self):
        thread.start_new_thread(self.handler,(self,))
    def send(self,d):
        self.data.append(d)
c = client()
c.sender()

