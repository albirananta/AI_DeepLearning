import torch
from ultralytics import YOLO

def main():
    device = 0 if torch.cuda.is_available() else 'cpu'
    model = YOLO("runs/detect/yolov12_test_gigi/weights/best.pt")
    
    print("=== MEMULAI EVALUASI MODEL ===")
    metrics = model.val(
        data="data gigi/data.yaml",
        split="val",
        imgsz=640,
        device=device
    )

if __name__ == "__main__":
    main()