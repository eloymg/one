import socket
import thread
from Crypto.Cipher import ARC4
from Crypto.Hash import SHA
from Crypto import Random


class server:
    def __init__(self, sock=None):
        self.data = []
        if sock is None:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        else:
            self.sock = sock
        self.__nonce = ""
        self.__nonce_recv = "n"

    def handler(self, _):
        sc, _ = self.sock.accept()
        while True:
            recibido = sc.recv(100)
            if recibido == '':
                sc.close()
                sc, _ = self.sock.accept()
            if recibido[:6] == 'nonce:':
                self.__nonce_recv = recibido[6:]
                print "Nonce recived: " + self.__nonce_recv
                sc.send("ACK")
            if recibido[:5] == 'data:':
                if self.__nonce_recv != self.__nonce:
                    cip = self.__cipher(self.__nonce_recv)
                    self.__nonce = self.__nonce_recv
                decrypted_data = cip.decrypt(recibido[5:])
                self.data.append(decrypted_data)
                print "data recived: " + decrypted_data
                sc.send("ACK")

    def server(self):
        self.sock.bind(("", 9999))
        self.sock.listen(1)
        thread.start_new_thread(self.handler, (self, ))

    def data(self):
        return self.data

    @staticmethod
    def __cipher(nonce):
        key = 'Very long and confidential key'
        tempkey = SHA.new(key + nonce).digest()
        cipher = ARC4.new(tempkey)
        return cipher