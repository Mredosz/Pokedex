from ultralytics import YOLO

# 1. Załaduj pretrenowany model
model = YOLO('yolov5s.pt')  # Możesz użyć 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt' dla większych modeli

# 2. Rozpocznij trening
model.train(
    data='../data/data.yaml',  # Plik YAML z konfiguracją zbioru danych
    epochs=1,         # Liczba epok
    imgsz=640,         # Rozmiar obrazu
    batch=16           # Rozmiar batcha
)

