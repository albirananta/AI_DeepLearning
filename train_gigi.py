import torch
from ultralytics import YOLO

def main():
    device = 0 if torch.cuda.is_available() else 'cpu'
    model = YOLO("yolov12n.pt")
    
    print("=== MEMULAI PROSES TRAINING YOLOv12 (DETEKSI GIGI) ===")
    results = model.train(
        data="data gigi/data.yaml",
        epochs=50,
        imgsz=640,
        batch=16,
        device=device,
        workers=2,
        name="yolov12_test_gigi",
        save=True,
        plots=True
    )
    print("\nTraining selesai!")

if __name__ == "__main__":
    main()