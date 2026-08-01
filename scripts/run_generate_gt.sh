set -e

# ---- EDIT THESE PATHS ----
YOLO_WEIGHTS="run/train/sonar_yolov11_m/weights/best.pt"  
SAM_CHECKPOINT="sam-hq/checkpoints/sam_hq_vit_l.pth" 
SAM_MODEL_TYPE="vit_l"                                
SOURCE="/Sonar_Dataset/images/test"
OUTPUT_ROOT="/Output/GT_MASK_samhq_"
TAG="yolov11_m_samhq"

CONF=0.20     
IOU=0.45
IMGSZ=640
DEVICE="cuda" 
# ---------------------------

python generate_gt_masks_samhq.py \
    --yolo-weights "${YOLO_WEIGHTS}" \
    --sam-checkpoint "${SAM_CHECKPOINT}" \
    --sam-model-type "${SAM_MODEL_TYPE}" \
    --source "${SOURCE}" \
    --output "${OUTPUT_ROOT}" \
    --tag "${TAG}" \
    --conf ${CONF} \
    --iou ${IOU} \
    --imgsz ${IMGSZ} \
    --device "${DEVICE}" \
    --save-polygons

echo ""
echo "Done. Results saved under: ${OUTPUT_ROOT}/${TAG}"
