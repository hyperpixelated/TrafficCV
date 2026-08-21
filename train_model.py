from ultralytics import YOLO

model = YOLO("yolov8n.pt")

print("Starting YOLO training session...")
results = model.train(
    data="data.yaml", 
    epochs=15, 
    imgsz=640, 
    plots=True 
)
print("Training complete! Your new weights are saved in the 'runs/detect/train/weights' folder.")
