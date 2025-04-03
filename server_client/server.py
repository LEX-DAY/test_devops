import socket
import pickle
import numpy as np
from modules import Model, draw_bboxes
import cv2



class Server:
    @staticmethod
    def start(host:int, port:int)->None:
        """
        Run server
        """
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((host,port))
            s.listen()
            print(f"Server started at {host}:{port}, await connection...")
            while True:
                conn, addr = s.accept()
                with conn:
                    data = conn.recv(1024)
                    if data:
                        image = inference_model()
                        serialized_image = pickle.dumps(image)
                        conn.sendall(len(serialized_image).to_bytes(8, 'big')) 
                        conn.sendall(serialized_image)
                        print(f"Message send to Client at {addr}!")

def inference_model():
    """
    Run inference model
    """
    PATH_MODEL = "./model/model.engine"
    IMG_PATH = "./test_img/1.png" 
    model = Model(PATH_MODEL, (640, 384))
    img  = cv2.imread(IMG_PATH)
    img = cv2.resize(img, (640, 384))
    boxes = model(img)
    frame = draw_bboxes(img, boxes)

    return frame

if __name__ == '__main__':
    HOST = '127.0.0.1'  
    PORT = 65432
    Server.start(HOST, PORT)



