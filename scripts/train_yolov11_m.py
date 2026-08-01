import os
import sys
from ultralytics import YOLO

# ============================================================
# Create output directory
# ============================================================
PROJECT_DIR = "/run/train"
RUN_NAME = "sonar_yolov11_m"

SAVE_DIR = os.path.join(PROJECT_DIR, RUN_NAME)
os.makedirs(SAVE_DIR, exist_ok=True)

# ============================================================
# Logger (prints to terminal + saves to file)
# ============================================================
class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w", buffering=1)

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

log_file = os.path.join(SAVE_DIR, "training.log")

logger = Logger(log_file)
sys.stdout = logger
sys.stderr = logger

print("=" * 70)
print("Starting YOLO11m Sonar Training")
print("=" * 70)

# ============================================================
# Load Model
# ============================================================
model = YOLO("yolo11m.pt")

# ============================================================
# Train
# ============================================================
results = model.train(

    data="/mnt/DATA/EE25M305/Sonar_Vision/Sonar_Dataset/data.yaml",

    epochs=100,
    batch=48,
    imgsz=640,
    patience=20,

    device=[0, 1, 2],
    workers=24,

    optimizer="auto",
    lr0=0.01,

    project=PROJECT_DIR,
    name=RUN_NAME,
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

print("\n")
print("=" * 70)
print("Training Completed Successfully")
print("=" * 70)
print(f"Training log saved to: {log_file}")