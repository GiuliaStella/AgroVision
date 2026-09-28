from ultralytics import YOLO

model = YOLO("yolo26n.pt")

model.train(
    data="data.yaml",
    epochs=100,
    imgsz=640,
    batch=4,
    patience=20,
    device="cpu",
    project="runs/detect",
    name="agrovision_100"
)