from ultralytics import YOLO

# Load YOLOv8 trained model
model = YOLO("runs/train/kuku/weights/best.pt")

# Export ke ONNX
model.export(format="onnx")  # hasil: runs/train/kuku/weights/best.onnx
print("✅ Exported to ONNX successfully!")
