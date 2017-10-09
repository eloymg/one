import socket
import thread

class server:
    def __init__(self, sock=None):
        self.data = []
        if sock is None:
            self.sock = socket.socket(
                socket.AF_INET, socket.SOCK_STREAM)
        else:
            self.sock = sock
    def handler(self,a):
        while True:
            sc , addr = self.sock.accept()
            recibido = sc.recv(20)
            if recibido.find("data:")>0:
                print recibido
                self.data.append(recibido)
                sc.send("ACK")
    def server(self):
       self.sock.bind(("", 9999))
       self.sock.listen(1)
       thread.start_new_thread(self.handler,(self,))
    def data(self):
        return self.data