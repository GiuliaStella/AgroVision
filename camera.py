import cv2
from ultralytics import YOLO

model = YOLO("runs/detect/agrovision/weights/best.pt")

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Não foi possível abrir a câmera.")
    exit()

print("AgroVision iniciado!")
print("Pressione Q para sair.")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Erro ao capturar imagem.")
        break

    results = model(frame, conf=0.70)

    frame_com_deteccoes = results[0].plot()

    cv2.imshow("AgroVision - YOLO26", frame_com_deteccoes)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()