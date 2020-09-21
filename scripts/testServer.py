# Echo client program
import socket
import time
import random

#
# Serveur envoyant un couple (DX , DY) représentant  les déplacements en X et Y que la pince du robot devra faire
# Cette infomaion est transmise sur solicitation du robot lorsqu'il envoie la commande 'ask'
#
# Usage : lancer d'abord ce programme puis le programme testClient qui se trouve sur le robot UR3
#

HOST = "10.31.64.202" # The remote host (IP du PC)
PORT = 30000 # The same port as used by the server

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s.bind((HOST, PORT)) # Bind to the port
s.listen(5) # Now wait for client connection.
c, addr = s.accept() # Establish connection with client.
while(True):
    try:
        msg = c.recv(1024)  # Réception d'un message depuis la socket en provenance du robot
        print(msg)
        time.sleep(1)
        if msg.decode() == "ask":  # On utilise decode car ce que l'on reçoit n'est pas une str mais un bytearray
            time.sleep(0.5)
            dx = random.randrange(100)  #   0 < dx < 100 mm
            dy = random.randrange(100)  #   0 < dy < 100 mm
            cmd = "({},{})".format(dx,dy)
            c.send(cmd.encode()) # Transmission (sous forme de bytearray) des déplacements en X et Y pour la pince du robot
            print(cmd)
    except socket.error as socketerror:
        print("erreur connexion socket")
c.close()
s.close()
print("Program finish")