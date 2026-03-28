from ultralytics import YOLO
import cv2

# caminho correto do modelo
model_path = r"C:\Users\Davi\Downloads\EPI DETECTION.v10i.yolov8\weights"

model = YOLO(model_path)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    annotated = results[0].plot()

    cv2.imshow("Detecção de EPI", annotated)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()