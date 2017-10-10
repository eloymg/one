import socket
import thread
import errno
import time
import sys


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
            try:
                sc , addr = self.sock.accept()
                recibido = sc.recv(100)
            except socket.error, e:
                err = e.args[0]
                if err == errno.EAGAIN or err == errno.EWOULDBLOCK:
                    time.sleep(1)
                    print 'No data available'
                    continue
                else:
                    # a "real" error occurred
                    print e
                    sys.exit(1)
            else:
                if recibido.find("data:")>0:
                    print recibido
                    self.data.append(recibido)
                    sc.send("ACK")
    def server(self):
       self.sock.setblocking(0)
       self.sock.bind(("", 9999))
       self.sock.listen(1)
       thread.start_new_thread(self.handler,(self,))
    def data(self):
        return self.data