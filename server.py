 
#importamos el modulo socket
import socket
 
#instanciamos un objeto para trabajar con el socket
s = socket.socket(socket.AF_INET)
#Con el metodo bind le indicamos que puerto debe escuchar y de que servidor esperar conexiones
#Es mejor dejarlo en blanco para recibir conexiones externas si es nuestro caso
s.bind(("", 9999))
 
#Aceptamos conexiones entrantes con el metodo listen, y ademas aplicamos como parametro
#El numero de conexiones entrantes que vamos a aceptar
s.listen(1)
 
#Instanciamos un objeto sc (socket cliente) para recibir datos, al recibir datos este 
#devolvera tambien un objeto que representa una tupla con los datos de conexion: IP y puerto
sc, addr = s.accept()
 
while True:
    while True:
        
        recibido = sc.recv(10)
        if recibido.find("/n")>0:
            print recibido
    sc, addr = s.accept()
 
#Cerramos la instancia del socket cliente y servidor
sc.close()
s.close()