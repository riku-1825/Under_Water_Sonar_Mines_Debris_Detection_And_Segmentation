from ultralytics import YOLO

model = YOLO("yolo11s.pt")

results = model.train(
    data="/Sonar_Dataset/data.yaml",

    epochs=100,
    batch=48,
    imgsz=640,
    patience=20,
    device=[0, 1, 2],
    workers=24,

    optimizer="auto",
    lr0=0.01,

    project="/run/train",
    name="sonar_yolov11_s",
    exist_ok=True,

    save=True,
    save_period=10,

    cache=True,

    augment=True,
    mosaic=0.3,
    mixup=0.0,
    copy_paste=0.0,

    hsv_h=0.0,
    hsv_s=0.0,
    hsv_v=0.0,

    degrees=5,
    translate=0.05,
    scale=0.20,
    shear=0.0,
    perspective=0.0,
    flipud=0.0,
    fliplr=0.5,

    plots=True,
    verbose=True
)