from ultralytics import YOLO
model = YOLO("yolo11m.pt")
train_results = model.train(
    data="data.yaml",
    epochs=150,  
    imgsz=640,
    device="cpu",           
)