import sys
import cv2
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QTimer
from modules.inference import TrtYOLO

class YOLODetector(QMainWindow):
    def __init__(self, model_path, source, imgsz=(640, 384)):
        super().__init__()
        self.model = TrtYOLO(model_path, imgsz)
        self.source = source
        self.imgsz = imgsz
        self.init_ui()
        self.setup_pipeline()

    def init_ui(self):
        self.setWindowTitle("YOLO Object Detection")
        self.setGeometry(100, 100, self.imgsz[0], self.imgsz[1])
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        
        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        central_widget.setLayout(layout)

    def setup_pipeline(self):
        if str(self.source).isdigit():
            self.cap = cv2.VideoCapture(int(self.source))
            self.timer = QTimer()
            self.timer.timeout.connect(self.process_frame)
            self.timer.start(30)
        else:
            self.process_image()

    def process_image(self):
        img = cv2.imread(self.source)
        if img is None:
            print(f"Error loading image: {self.source}")
            return
            
        img = cv2.resize(img, self.imgsz)
        detections = self.model(img)
        self.display_result(img, detections)

    def process_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            self.timer.stop()
            return
            
        frame = cv2.resize(frame, self.imgsz)
        detections = self.model(frame)
        self.display_result(frame, detections)

    def draw_boxes(self, img, boxes):
        for box in boxes:
            x1, y1, x2, y2, conf, cls = box
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(img, f"{cls}: {conf:.2f}", (int(x1), int(y1)-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        return img

    def display_result(self, img, detections):
        img = self.draw_boxes(img, detections)
        qt_img = QImage(img.data, img.shape[1], img.shape[0], 
                       QImage.Format_RGB888).rgbSwapped()
        self.image_label.setPixmap(QPixmap.fromImage(qt_img))

    def closeEvent(self, event):
        if hasattr(self, 'cap'):
            self.cap.release()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    model_path = "./model/model.engine"
    source = "./test_img/1.png"  # или 0 для веб-камеры
    imgsz = (640, 384)
    
    window = YOLODetector(model_path, source, imgsz)
    window.show()
    sys.exit(app.exec_())