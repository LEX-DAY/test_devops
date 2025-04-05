import cv2
import sys
from modules.inference import TrtYOLO

def draw_bboxes(img, boxes):
    """ функция для отрисовки боксов на изображении"""
    for box in boxes:
        x1, y1, x2, y2, conf, cls = box
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        label = f"{cls}: {conf:.2f}"
        cv2.putText(img, label, (int(x1), int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return img

if __name__ == "__main__":
    PATH_MODEL = "./model/model.engine"
    IMG_PATH = "./test_img/1.png"
    
    # Загружаем модель
    model = TrtYOLO(PATH_MODEL, (640, 384))
    
    # Читаем изображение
    img = cv2.imread(IMG_PATH)
    if img is None:
        print(f"Ошибка: Не удалось загрузить изображение {IMG_PATH}")
        sys.exit(1)
    
    img = cv2.resize(img, (640, 384))
    
    # Запускаем инференс
    boxes = model(img)
    
    # Отрисовываем боксы
    frame = draw_bboxes(img, boxes)
    
    # Отображаем результат
    cv2.imshow("YOLO Inference", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()