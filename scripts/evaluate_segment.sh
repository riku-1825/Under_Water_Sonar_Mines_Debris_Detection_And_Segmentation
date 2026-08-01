set -e

# ---- EDIT THESE PATHS ----
PRED_N="Output/Segment_sam2/yolov11_n/semantic"
PRED_S="Output/Segment_sam2/yolov11_s/semantic"
PRED_M="Output/Segment_sam2/yolov11_m/semantic"

GT_N="Output/GT_MASK_samhq_1/yolov11_n_samhq/masks_semantic"
GT_S="Output/GT_MASK_samhq_1/yolov11_s_samhq/masks_semantic"
GT_M="Output/GT_MASK_samhq_1/yolov11_m_samhq/masks_semantic"

DATA_YAML="/Sonar_Dataset/data.yaml"
OUTPUT_ROOT="/Output/Evaluate_Segment"
DEVICES="cuda:0,cuda:1,cuda:2"
# ---------------------------

python evaluate_segment.py \
    --pred-dirs "${PRED_N}" "${PRED_S}" "${PRED_M}" \
    --gt-dirs   "${GT_N}"   "${GT_S}"   "${GT_M}" \
    --tags yolov11_n yolov11_s yolov11_m \
    --data "${DATA_YAML}" \
    --devices "${DEVICES}" \
    --output "${OUTPUT_ROOT}"

echo ""
echo "Done. Results saved under: ${OUTPUT_ROOT}"
