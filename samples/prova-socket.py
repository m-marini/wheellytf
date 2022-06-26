import socket
import mmap

HOST = "192.168.1.11"
PORT = 22

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    print("Connecting...")
    s.connect((HOST, PORT))
    print("Connected.")
    #s.sendall(b"mv 90 0\n")
    x = s.makefile("rb")
    while True:
        try:
            data = x.readline()
            if data != None:
                print("Received", data.decode("utf-8"))
        except OSError as msg:
            if msg.errno != 11:
                print(msg)



