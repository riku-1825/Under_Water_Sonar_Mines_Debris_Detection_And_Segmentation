set -e

# ---- EDIT THESE PATHS ----
WEIGHTS_N="/run/train/sonar_yolov11_n/weights/best.pt"
WEIGHTS_S="/run/train/sonar_yolov11_s/weights/best.pt"
WEIGHTS_M="/run/train/sonar_yolov11_m/weights/best.pt"

SOURCE="/Sonar_Dataset/images/test"  # folder of test images
DATA_YAML="/Sonar_Dataset/data.yaml"        
OUTPUT_ROOT="/Output/Detection"  # where to save results

CONF=0.25
IOU=0.45
IMGSZ=640
DEVICE="0,1,2"                         # single GPU: "0" | multi-GPU: "0,1,2" | CPU: "cpu"
WORKERS=24                             # dataloader workers
# ---------------------------

run_model () {
    local weights=$1
    local tag=$2

    echo ""
    echo "=================================================="
    echo " Running inference: ${tag}  (${weights})"
    echo "=================================================="

    DATA_ARG=""
    if [ -n "${DATA_YAML}" ]; then
        DATA_ARG="--data ${DATA_YAML}"
    fi

    python detect_main.py \
        --weights "${weights}" \
        --model-tag "${tag}" \
        --source "${SOURCE}" \
        --output "${OUTPUT_ROOT}" \
        ${DATA_ARG} \
        --conf ${CONF} \
        --iou ${IOU} \
        --imgsz ${IMGSZ} \
        --device "${DEVICE}" \
        --workers ${WORKERS}
}

run_model "${WEIGHTS_N}" "yolov11_n"
run_model "${WEIGHTS_S}" "yolov11_s"
run_model "${WEIGHTS_M}" "yolov11_m"

echo ""
echo "All done. Results saved under: ${OUTPUT_ROOT}"
