python segment_sam2.py \
    --images /Sonar_Dataset/images/test \
    --yolo /run/train/sonar_yolov11_n/weights/best.pt \
    --sam-config sam2.1/sam2.1_hiera_s.yaml \
    --sam-checkpoint /sam2/checkpoints/sam2.1_hiera_small.pt \
    --output /Output/Segment_sam2/yolov11_n
