import socket
import thread
import errno
import time
import sys


class server:
    def __init__(self, sock=None):
        self.data = []
        if sock is None:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        else:
            self.sock = sock
    def handler(self, _):
        sc, _ = self.sock.accept()
        while True:
            recibido = sc.recv(100)
            if recibido == '':
                sc.close()
                sc, _ = self.sock.accept()
            if recibido.find('nonce:') > 0:
                self.__nonce = recibido.split(":")[1]
                print "Nonce recived: "+self.__nonce
                sc.send("ACK")
    def server(self):
        self.sock.bind(("", 9999))
        self.sock.listen(1)
        thread.start_new_thread(self.handler, (self, ))
    def data(self):
        return self.data


s = server()
s.server()
while True:
    a = 2
