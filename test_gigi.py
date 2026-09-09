import torch
from ultralytics import YOLO

def main():
    device = 0 if torch.cuda.is_available() else 'cpu'
    model = YOLO("runs/detect/yolov12_test_gigi/weights/best.pt")
    
    print("=== MEMULAI PREDIKSI GAMBAR TEST ===")
    results = model.predict(
        source="data gigi/test/image",
        conf=0.5,
        device=device,
        save=True,
        project="runs/detect",
        name="hasil_test_prediksi"
    )
    print("\nPrediksi selesai! Hasil ada di: runs/detect/hasil_test_prediksi")

if __name__ == "__main__":
    main()