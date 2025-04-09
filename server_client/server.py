import socket
import pickle
import numpy as np
import cv2
from modules import Model, draw_bboxes


class Server:
    @staticmethod
    def start(host: str, port: int) -> None:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((host, port))
            s.listen()
            print(f"🚀 Server started at {host}:{port}, waiting for connections...")

            while True:
                conn, addr = s.accept()
                with conn:
                    print(f"📞 Connection from {addr}")

                    try:
                        request = conn.recv(1024)
                        if not request:
                            print("⚠️ Empty request received.")
                            continue

                        command = request.decode().strip()
                        if command == "Generate image":
                            print("🧠 Generating image...")
                            image = inference_model()

                            # Сериализация и отправка
                            serialized_image = pickle.dumps(image)
                            conn.sendall(len(serialized_image).to_bytes(8, 'big'))
                            conn.sendall(serialized_image)
                            print(f"✅ Image sent to client at {addr}")

                        else:
                            print(f"⚠️ Unknown command: {command}")

                    except Exception as e:
                        print(f"❌ Error during handling request from {addr}: {e}")


def inference_model():
    """
    Run inference using TensorRT model.
    """
    PATH_MODEL = "./model/model.engine"
    IMG_PATH = "./test_img/1.png"

    # Проверка наличия файла
    if not os.path.exists(PATH_MODEL):
        raise FileNotFoundError(f"Model not found: {PATH_MODEL}")
    if not os.path.exists(IMG_PATH):
        raise FileNotFoundError(f"Image not found: {IMG_PATH}")

    model = Model(PATH_MODEL, (640, 384))

    img = cv2.imread(IMG_PATH)
    img = cv2.resize(img, (640, 384))
    boxes = model(img)
    result = draw_bboxes(img, boxes)

    return result


if __name__ == '__main__':
    import os
    HOST = '0.0.0.0'  # Принимаем подключения извне, если нужно
    PORT = 65432
    Server.start(HOST, PORT)
