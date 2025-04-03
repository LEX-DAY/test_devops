import socket
import pickle
import numpy as np
import cv2


class Client:
    @staticmethod
    def send_request(host:int, port:int)-> None:
        """
        Send request to server to generate image
        """
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((host, port))
            s.sendall(b"Generate image")
            print("Send to server")
            data_size = int.from_bytes(s.recv(8), 'big')
            data = b""
            while len(data) < data_size:
                packet = s.recv(data_size - len(data))
                if not packet:
                    break
                data += packet

            image = pickle.loads(data)

            cv2.imshow("Received Image", image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()  

if __name__ == "__main__":
    Client.send_request('127.0.0.1', 65432)
