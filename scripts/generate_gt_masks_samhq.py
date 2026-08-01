import argparse
import csv
import json
import logging
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np

try:
    from ultralytics import YOLO
except ImportError:
    sys.exit("ultralytics not installed. Run: pip install ultralytics")

try:
    from segment_anything_hq import sam_model_registry, SamPredictor
except ImportError:
    sys.exit("segment-anything-hq not installed. Run: pip install segment-anything-hq  "
              "(see setup_samhq.sh)")

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
BOX_COLOR = (0, 255, 0)
TEXT_COLOR = (0, 255, 0)
FONT = cv2.FONT_HERSHEY_SIMPLEX

# distinct BGR colors cycled per class for mask fill / overlay
PALETTE = [
    (255, 0, 0), (0, 0, 255), (0, 255, 255), (255, 0, 255), (255, 128, 0),
    (0, 128, 255), (128, 0, 255), (0, 255, 128), (128, 255, 0), (255, 255, 0),
    (0, 128, 128),
]


def parse_args():
    p = argparse.ArgumentParser(description="YOLO-box-prompted SAM-HQ pseudo-GT mask generation")
    p.add_argument("--yolo-weights", required=True, help="Path to trained YOLOv11 best.pt")
    p.add_argument("--sam-checkpoint", required=True, help="Path to SAM-HQ .pth checkpoint")
    p.add_argument("--sam-model-type", default="vit_l", choices=["vit_h", "vit_l", "vit_b", "vit_tiny"])
    p.add_argument("--source", required=True, help="Folder of test images")
    p.add_argument("--output", default="Output/GT_Masks", help="Root output directory")
    p.add_argument("--tag", required=True, help="Folder name for this run, e.g. yolov11_n_samhq")
    p.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold")
    p.add_argument("--iou", type=float, default=0.45, help="YOLO NMS IoU threshold")
    p.add_argument("--imgsz", type=int, default=640, help="YOLO inference image size")
    p.add_argument("--device", default="cuda", help="'cuda', '0', '0,1,2', or 'cpu' (applies to both models)")
    p.add_argument("--mask-alpha", type=float, default=0.45, help="Overlay transparency for mask fill")
    p.add_argument("--save-polygons", action="store_true",
                    help="Also save YOLO-seg polygon labels (class_id x1 y1 x2 y2 ...)")
    p.add_argument("--min-contour-area", type=int, default=15,
                    help="Discard polygon contours smaller than this (noise removal)")
    return p.parse_args()


def setup_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger(log_path.stem + "_gt")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fh = logging.FileHandler(log_path, mode="w")
    fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S"))
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def colorize_semantic(semantic_mask: np.ndarray, palette) -> np.ndarray:
    """Render a class-id semantic mask (0 = background) as a flat-color BGR image,
    cycling through PALETTE the same way the instance-fill overlay does."""
    h, w = semantic_mask.shape
    color_img = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id in np.unique(semantic_mask):
        if class_id == 0:
            continue
        color = palette[(int(class_id) - 1) % len(palette)]
        color_img[semantic_mask == class_id] = color
    return color_img


def mask_to_yolo_polygons(binary_mask: np.ndarray, img_w: int, img_h: int, min_area: int):
    """Return list of normalized polygon coordinate lists (one per contour) from a binary mask."""
    contours, _ = cv2.findContours(binary_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for c in contours:
        if cv2.contourArea(c) < min_area:
            continue
        c = c.reshape(-1, 2)
        norm = []
        for x, y in c:
            norm.append(round(x / img_w, 6))
            norm.append(round(y / img_h, 6))
        polygons.append(norm)
    return polygons


def main():
    args = parse_args()

    out_root = Path(args.output) / args.tag
    dirs = {
        "masks_binary": out_root / "masks_binary",
        "masks_semantic": out_root / "masks_semantic",
        "semantic_color": out_root / "semantic_color",
        "masks_instances": out_root / "masks_instances",
        "overlays": out_root / "overlays",
        "json": out_root / "json",
        "labels": out_root / "labels",
    }
    if args.save_polygons:
        dirs["labels_seg"] = out_root / "labels_seg"
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(out_root / "generation.log")
    logger.info(f"=== Pseudo-GT Mask Generation: {args.tag} ===")
    logger.info(f"YOLO weights : {args.yolo_weights}")
    logger.info(f"SAM-HQ ckpt  : {args.sam_checkpoint}  (type={args.sam_model_type})")
    logger.info(f"Source       : {args.source}")
    logger.info(f"conf={args.conf}  iou={args.iou}  imgsz={args.imgsz}  device={args.device}  "
                f"save_polygons={args.save_polygons}")

    # ---- load models ----
    yolo = YOLO(args.yolo_weights)
    logger.info(f"YOLO loaded. Classes: {yolo.names}")

    sam = sam_model_registry[args.sam_model_type](checkpoint=args.sam_checkpoint)
    sam.to(args.device)
    predictor = SamPredictor(sam)
    logger.info("SAM-HQ loaded.")

    src_dir = Path(args.source)
    image_paths = sorted([p for p in src_dir.iterdir() if p.suffix.lower() in IMG_EXTS])
    if not image_paths:
        logger.error(f"No images found in {src_dir}")
        sys.exit(1)
    logger.info(f"Found {len(image_paths)} test images.")

    per_image_times = []
    per_instance_scores = []
    class_counter = {}
    total_instances = 0
    images_with_no_detection = 0

    for idx, img_path in enumerate(image_paths, 1):
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            logger.warning(f"Could not read {img_path.name}, skipping.")
            continue
        h, w = img_bgr.shape[:2]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        t0 = time.time()

        # ---- YOLO detection ----
        det = yolo.predict(img_bgr, conf=args.conf, iou=args.iou, imgsz=args.imgsz,
                            device=args.device, verbose=False)[0]

        boxes_data = []
        if det.boxes is not None and len(det.boxes) > 0:
            xyxy = det.boxes.xyxy.cpu().numpy()
            confs = det.boxes.conf.cpu().numpy()
            clss = det.boxes.cls.cpu().numpy()
            boxes_data = list(zip(xyxy, confs, clss))

        if not boxes_data:
            images_with_no_detection += 1
            elapsed_ms = (time.time() - t0) * 1000.0
            per_image_times.append(elapsed_ms)
            empty_semantic = np.zeros((h, w), dtype=np.uint8)
            cv2.imwrite(str(dirs["masks_binary"] / img_path.name), np.zeros((h, w), dtype=np.uint8))
            cv2.imwrite(str(dirs["masks_semantic"] / img_path.name), empty_semantic)
            cv2.imwrite(str(dirs["semantic_color"] / img_path.name), colorize_semantic(empty_semantic, PALETTE))
            cv2.imwrite(str(dirs["overlays"] / img_path.name), img_bgr)
            with open(dirs["json"] / f"{img_path.stem}.json", "w") as f:
                json.dump([], f, indent=2)
            (dirs["labels"] / f"{img_path.stem}.txt").write_text("")
            if args.save_polygons:
                (dirs["labels_seg"] / f"{img_path.stem}.txt").write_text("")
            logger.info(f"[{idx}/{len(image_paths)}] {img_path.name} -> 0 detections | {elapsed_ms:.1f} ms")
            continue

        # ---- SAM-HQ box-prompted segmentation ----
        predictor.set_image(img_rgb)

        binary_mask = np.zeros((h, w), dtype=np.uint8)
        semantic_mask = np.zeros((h, w), dtype=np.uint8)
        overlay = img_bgr.copy()
        color_layer = np.zeros_like(img_bgr)
        det_label_lines = []
        seg_label_lines = []
        json_items = []

        for inst_idx, ((x1, y1, x2, y2), conf, cls_id) in enumerate(boxes_data):
            cls_id = int(cls_id)
            cls_name = yolo.names[cls_id]
            box_np = np.array([x1, y1, x2, y2])

            masks, scores, _ = predictor.predict(box=box_np, multimask_output=False)
            m = masks[0].astype(np.uint8)       # HxW, 0/1
            score = float(scores[0])
            per_instance_scores.append(score)

            binary_mask[m == 1] = 255
            semantic_mask[m == 1] = cls_id + 1

            color = PALETTE[cls_id % len(PALETTE)]
            color_layer[m == 1] = color

            mask_name = f"{img_path.stem}_{inst_idx}.png"
            cv2.imwrite(str(dirs["masks_instances"] / mask_name), m * 255)

            x1i, y1i, x2i, y2i = map(int, [x1, y1, x2, y2])
            json_items.append({
                "bbox": [x1i, y1i, x2i, y2i],
                "class": cls_id,
                "yolo_conf": float(conf),
                "sam_score": score,
                "mask": mask_name,
            })

            xc = ((x1 + x2) / 2) / w
            yc = ((y1 + y2) / 2) / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h
            det_label_lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

            if args.save_polygons:
                for poly in mask_to_yolo_polygons(m, w, h, args.min_contour_area):
                    seg_label_lines.append(f"{cls_id} " + " ".join(f"{v:.6f}" for v in poly))

            class_counter[cls_name] = class_counter.get(cls_name, 0) + 1
            total_instances += 1

        # blend mask fill once, then draw boxes+labels on top
        overlay = cv2.addWeighted(color_layer, args.mask_alpha, overlay, 1 - args.mask_alpha, 0)
        overlay = np.where(color_layer > 0, overlay, img_bgr)  # keep untouched pixels sharp
        for (x1, y1, x2, y2), conf, cls_id in boxes_data:
            x1i, y1i, x2i, y2i = map(int, [x1, y1, x2, y2])
            cls_name = yolo.names[int(cls_id)]
            label = f"{cls_name} {conf:.2f}"
            cv2.rectangle(overlay, (x1i, y1i), (x2i, y2i), BOX_COLOR, 2)
            (tw, th), baseline = cv2.getTextSize(label, FONT, 0.6, 2)
            ty = max(y1i - 8, th + 4)
            cv2.rectangle(overlay, (x1i, ty - th - baseline), (x1i + tw, ty + baseline), (0, 0, 0), -1)
            cv2.putText(overlay, label, (x1i, ty), FONT, 0.6, TEXT_COLOR, 2, cv2.LINE_AA)

        elapsed_ms = (time.time() - t0) * 1000.0
        per_image_times.append(elapsed_ms)

        cv2.imwrite(str(dirs["masks_binary"] / img_path.name), binary_mask)
        cv2.imwrite(str(dirs["masks_semantic"] / img_path.name), semantic_mask)
        cv2.imwrite(str(dirs["semantic_color"] / img_path.name), colorize_semantic(semantic_mask, PALETTE))
        cv2.imwrite(str(dirs["overlays"] / img_path.name), overlay)
        with open(dirs["json"] / f"{img_path.stem}.json", "w") as f:
            json.dump(json_items, f, indent=2)
        (dirs["labels"] / f"{img_path.stem}.txt").write_text("\n".join(det_label_lines) + "\n")
        if args.save_polygons:
            (dirs["labels_seg"] / f"{img_path.stem}.txt").write_text("\n".join(seg_label_lines) + "\n")

        logger.info(f"[{idx}/{len(image_paths)}] {img_path.name} -> {len(boxes_data)} instance(s) | "
                    f"{elapsed_ms:.1f} ms")

    # ---- summary.txt ----
    total_images = len(image_paths)
    avg_time = statistics.mean(per_image_times) if per_image_times else 0.0
    avg_fps = statistics.mean([1000.0 / t for t in per_image_times if t > 0]) if per_image_times else 0.0
    avg_score = statistics.mean(per_instance_scores) if per_instance_scores else 0.0
    low_conf_masks = sum(1 for s in per_instance_scores if s < 0.85)

    lines = [
        f"Pseudo-GT Mask Generation Summary — {args.tag}",
        "=" * 55,
        f"YOLO weights            : {args.yolo_weights}",
        f"SAM-HQ checkpoint        : {args.sam_checkpoint} ({args.sam_model_type})",
        f"Source                   : {args.source}",
        f"Total images             : {total_images}",
        f"Images with 0 detections : {images_with_no_detection}",
        f"Total mask instances     : {total_instances}",
        f"Avg SAM-HQ mask score    : {avg_score:.4f}",
        f"Instances w/ score < 0.85: {low_conf_masks}  (inspect these — likely weak masks)",
        f"Avg time / image (ms)    : {avg_time:.2f}",
        f"Avg FPS                  : {avg_fps:.2f}",
        "",
        "Instances per class:",
    ]
    for cls_name, count in sorted(class_counter.items(), key=lambda x: -x[1]):
        lines.append(f"  {cls_name:<20}: {count}")
    lines += [
        
        "",
        "IMPORTANT: these are auto-generated pseudo-ground-truth masks (SAM-HQ output",
        "prompted by YOLO boxes), not human-verified ground truth. State this clearly",
        "when reporting IoU/Dice computed against them.",
        "",
        "Folders:",
        "  masks_binary/    0/255 single-channel binary mask",
        "  masks_semantic/  pixel value = class_id + 1 (0 = background)",
        "  semantic_color/  semantic mask rendered with one flat color per class",
        "  masks_instances/ one 0/255 mask file per detected object (<stem>_<idx>.png)",
        "  overlays/        mask fill + box + class/confidence label",
        "  json/            per-image list of {bbox, class, yolo_conf, sam_score, mask}",
        "  labels/          class_id x_center y_center width height (normalized)",
    ]
    if args.save_polygons:
        lines.append("  labels_seg/      class_id x1 y1 x2 y2 ... (normalized polygon, YOLO-seg format)")

    (out_root / "summary.txt").write_text("\n".join(lines) + "\n")
    logger.info(f"Saved {out_root / 'summary.txt'}")
    logger.info("=== Generation complete ===")


if __name__ == "__main__":
    main()
