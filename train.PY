from ultralytics import YOLO

model = YOLO("yolo26n.pt")

model.train(
    data="data.yaml",
    epochs=50,
    imgsz=640,
    batch=4,
    device="cpu",
    workers=2,
    patience=20,
    name="agrovision"
)