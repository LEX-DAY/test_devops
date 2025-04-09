import socket
import pickle
import numpy as np
import cv2


class Client:
    @staticmethod
    def send_request(host: str, port: int) -> None:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.connect((host, port))
                s.sendall(b"Generate image")
                print("Sent request to server.")

                # Получаем размер данных (8 байт)
                data_size_bytes = s.recv(8)
                if len(data_size_bytes) < 8:
                    raise ValueError("Failed to receive image size from server.")

                data_size = int.from_bytes(data_size_bytes, 'big')
                print(f"Expecting {data_size} bytes of image data.")

                # Читаем данные полностью
                data = b""
                while len(data) < data_size:
                    packet = s.recv(min(4096, data_size - len(data)))
                    if not packet:
                        raise ConnectionError("Connection lost while receiving data.")
                    data += packet

                # Десериализация
                image = pickle.loads(data)

                # Проверка, что это изображение
                if not isinstance(image, np.ndarray):
                    raise ValueError("Received data is not a valid image.")

                # Отображение
                cv2.imshow("Received Image", image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

        except ConnectionRefusedError:
            print("❌ Connection refused. Make sure the server is running.")
        except Exception as e:
            print(f"❌ Error: {e}")


if __name__ == "__main__":
    Client.send_request('127.0.0.1', 65432)
