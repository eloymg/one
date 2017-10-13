import socket
import thread
from Crypto.Cipher import ARC4
from Crypto.Hash import SHA
from Crypto import Random
"""
for i in intensity_vec:
    cipher_msg.append(cipher1.encrypt(str(i)))
"""


class client(object):
    def __init__(self):
        self.data = []

    def handler(self, _):
        nonce_counter = 101git
        while True:
            if nonce_counter > 100:
                cipher, nonce = self.__cipher()
                response = self.send('nonce:'+nonce)
                if response == "timeout":
                    print "timeout!"
                elif response == "ACK":
                    nonce_counter = 0
                else:
                    print "bad data"
          
            if len(self.data) > 0:
                data = self.data[0]
                response = self.send("data:"+cipher.encrypt(data))
                if response == "timeout":
                    print "timeout!"
                elif response == "ACK":
                    self.data.pop(0)
                    nonce_counter += 1
                else:
                    print "bad data"

    def sender(self):
        thread.start_new_thread(self.handler, (self, ))

    def buffer(self, data):
        """Append data for send"""
        self.data.append(data)

    @staticmethod
    def send(data):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect(("127.0.0.1", 9999))
        sock.send(data)
        sock.settimeout(5)
        try:
            response = sock.recv(1024)
        except socket.timeout:
            response = "timeout"
        sock.close()
        return response
    @staticmethod
    def __cipher():
        key = 'Very long and confidential key'
        nonce = Random.new().read(16)
        tempkey = SHA.new(key + nonce).digest()
        cipher = ARC4.new(tempkey)
        return cipher, nonce




