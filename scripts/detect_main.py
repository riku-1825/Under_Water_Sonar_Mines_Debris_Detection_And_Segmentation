import argparse
import csv
import logging
import statistics
import sys
import time
from pathlib import Path

import cv2

try:
    from ultralytics import YOLO
except ImportError:
    sys.exit("ultralytics not installed. Run: pip install ultralytics")

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
BOX_COLOR = (0, 255, 0)      # green, BGR
TEXT_COLOR = (0, 255, 0)     # green
FONT = cv2.FONT_HERSHEY_SIMPLEX


def parse_args():
    p = argparse.ArgumentParser(description="YOLOv11 test-set inference + logging pipeline")
    p.add_argument("--weights", required=True, help="Path to trained .pt weights (best.pt)")
    p.add_argument("--model-tag", required=True, help="Folder name for this model, e.g. yolov11_n")
    p.add_argument("--source", required=True, help="Folder of test images")
    p.add_argument("--output", default="Output/Detection", help="Root output directory")
    p.add_argument("--data", default=None, help="Optional data.yaml for computing P/R/mAP via validation")
    p.add_argument("--split", default="test", help="Split to use for validation metrics (test/val)")
    p.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    p.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    p.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    p.add_argument("--device", default="", help="cuda device(s), e.g. 0 or '0,1,2' for multi-GPU or 'cpu'")
    p.add_argument("--workers", type=int, default=8, help="Dataloader workers (used for --data validation pass)")
    p.add_argument("--line-thickness", type=int, default=2, help="Bounding box line thickness")
    return p.parse_args()


def setup_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger(log_path.stem)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fh = logging.FileHandler(log_path, mode="w")
    fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S"))
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(logging.Formatter("%(message)s"))

    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def get_class_level_metrics(model: YOLO, data_yaml: str, split: str, imgsz: int, device: str, workers: int, logger: logging.Logger):
    """Run model.val() and return {class_name: {precision, recall, map50, map5095}} plus overall dict.
    Returns (per_class_dict, overall_dict). Falls back to 'N/A' fields on any API mismatch."""
    per_class, overall = {}, {"precision": "N/A", "recall": "N/A", "map50": "N/A", "map5095": "N/A"}
    if not data_yaml:
        logger.info("No --data provided: skipping Precision/Recall/mAP computation (val metrics = N/A).")
        return per_class, overall

    try:
        logger.info(f"Running validation on split='{split}' using {data_yaml} for P/R/mAP ...")
        metrics = model.val(data=data_yaml, split=split, imgsz=imgsz, device=device, workers=workers, verbose=False)
        box = metrics.box

        overall = {
            "precision": round(float(box.mp), 4),
            "recall": round(float(box.mr), 4),
            "map50": round(float(box.map50), 4),
            "map5095": round(float(box.map), 4),
        }

        class_indices = list(getattr(box, "ap_class_index", []))
        names = model.names
        p_arr, r_arr = box.p, box.r
        ap50_arr = box.ap50 if hasattr(box, "ap50") else None
        ap_arr = box.maps if hasattr(box, "maps") else None

        for i, cls_idx in enumerate(class_indices):
            cls_name = names[int(cls_idx)]
            per_class[cls_name] = {
                "precision": round(float(p_arr[i]), 4) if p_arr is not None else "N/A",
                "recall": round(float(r_arr[i]), 4) if r_arr is not None else "N/A",
                "map50": round(float(ap50_arr[i]), 4) if ap50_arr is not None else "N/A",
                "map5095": round(float(ap_arr[int(cls_idx)]), 4) if ap_arr is not None else "N/A",
            }
        logger.info(f"Validation complete. Overall: {overall}")
    except Exception as e:
        logger.warning(f"Could not compute validation metrics ({e}). Precision/Recall/mAP will be 'N/A'.")
    return per_class, overall


def draw_detections(img, boxes, names):
    for x1, y1, x2, y2, conf, cls_id in boxes:
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        cls_name = names[int(cls_id)]
        label = f"{cls_name} {conf:.2f}"

        cv2.rectangle(img, (x1, y1), (x2, y2), BOX_COLOR, 2)
        (tw, th), baseline = cv2.getTextSize(label, FONT, 0.6, 2)
        text_y = max(y1 - 8, th + 4)
        cv2.rectangle(img, (x1, text_y - th - baseline), (x1 + tw, text_y + baseline), (0, 0, 0), -1)
        cv2.putText(img, label, (x1, text_y), FONT, 0.6, TEXT_COLOR, 2, cv2.LINE_AA)
    return img


def main():
    args = parse_args()

    out_root = Path(args.output) / args.model_tag
    img_out_dir = out_root / "images"
    out_root.mkdir(parents=True, exist_ok=True)
    img_out_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(out_root / "inference.log")
    logger.info(f"=== YOLOv11 Inference Run: {args.model_tag} ===")
    logger.info(f"Weights: {args.weights}")
    logger.info(f"Source images: {args.source}")
    logger.info(f"conf={args.conf}  iou={args.iou}  imgsz={args.imgsz}  device='{args.device}'  workers={args.workers}")

    model = YOLO(args.weights)
    logger.info(f"Model loaded. Classes: {model.names}")

    per_class_metrics, overall_metrics = get_class_level_metrics(
        model, args.data, args.split, args.imgsz, args.device, args.workers, logger
    )

    src_dir = Path(args.source)
    image_paths = sorted([p for p in src_dir.iterdir() if p.suffix.lower() in IMG_EXTS])
    if not image_paths:
        logger.error(f"No images found in {src_dir}")
        sys.exit(1)
    logger.info(f"Found {len(image_paths)} test images.")

    detections_rows = []   # image, class, confidence, x1, y1, x2, y2
    metrics_rows = []      # image, object, precision, recall, map50, map5095, inference_time_ms, fps
    per_image_times = []
    per_image_det_counts = []
    class_counter = {}

    for idx, img_path in enumerate(image_paths, 1):
        img = cv2.imread(str(img_path))
        if img is None:
            logger.warning(f"Could not read image: {img_path.name}, skipping.")
            continue

        t0 = time.time()
        results = model.predict(
            img, conf=args.conf, iou=args.iou, imgsz=args.imgsz, device=args.device,
            workers=args.workers, verbose=False
        )
        elapsed_ms = (time.time() - t0) * 1000.0
        fps = 1000.0 / elapsed_ms if elapsed_ms > 0 else 0.0
        per_image_times.append(elapsed_ms)

        r = results[0]
        boxes_data = []
        if r.boxes is not None and len(r.boxes) > 0:
            xyxy = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            clss = r.boxes.cls.cpu().numpy()
            for (x1, y1, x2, y2), conf, cls_id in zip(xyxy, confs, clss):
                boxes_data.append((x1, y1, x2, y2, conf, cls_id))

        per_image_det_counts.append(len(boxes_data))

        annotated = draw_detections(img.copy(), boxes_data, model.names)
        cv2.imwrite(str(img_out_dir / img_path.name), annotated)

        if boxes_data:
            for x1, y1, x2, y2, conf, cls_id in boxes_data:
                cls_name = model.names[int(cls_id)]
                class_counter[cls_name] = class_counter.get(cls_name, 0) + 1

                detections_rows.append([
                    img_path.name, cls_name, round(float(conf), 4),
                    round(float(x1), 1), round(float(y1), 1), round(float(x2), 1), round(float(y2), 1)
                ])

                cm = per_class_metrics.get(cls_name, {"precision": "N/A", "recall": "N/A", "map50": "N/A", "map5095": "N/A"})
                metrics_rows.append([
                    img_path.name, cls_name, cm["precision"], cm["recall"], cm["map50"], cm["map5095"],
                    round(elapsed_ms, 2), round(fps, 2)
                ])
            logger.info(f"[{idx}/{len(image_paths)}] {img_path.name} -> {len(boxes_data)} detection(s) | "
                        f"{elapsed_ms:.1f} ms | {fps:.1f} FPS")
        else:
            metrics_rows.append([img_path.name, "none", "N/A", "N/A", "N/A", "N/A", round(elapsed_ms, 2), round(fps, 2)])
            logger.info(f"[{idx}/{len(image_paths)}] {img_path.name} -> 0 detections | {elapsed_ms:.1f} ms | {fps:.1f} FPS")

    # ---- write detections.csv ----
    det_csv_path = out_root / "detections.csv"
    with open(det_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "class", "confidence", "x1", "y1", "x2", "y2"])
        writer.writerows(detections_rows)
    logger.info(f"Saved {det_csv_path} ({len(detections_rows)} detection rows).")

    # ---- write metrics.csv ----
    metrics_csv_path = out_root / "metrics.csv"
    with open(metrics_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "object", "precision", "recall", "map50", "map50_95", "inference_time_ms", "fps"])
        writer.writerows(metrics_rows)
    logger.info(f"Saved {metrics_csv_path} ({len(metrics_rows)} rows).")

    # ---- summary.txt ----
    total_images = len(image_paths)
    total_detections = sum(per_image_det_counts)
    avg_time = statistics.mean(per_image_times) if per_image_times else 0.0
    avg_fps = statistics.mean([1000.0 / t for t in per_image_times if t > 0]) if per_image_times else 0.0
    images_with_no_detection = sum(1 for c in per_image_det_counts if c == 0)

    summary_lines = [
        f"YOLOv11 Inference Summary — {args.model_tag}",
        "=" * 50,
        f"Weights                 : {args.weights}",
        f"Source                  : {args.source}",
        f"Total images            : {total_images}",
        f"Total detections        : {total_detections}",
        f"Images with 0 detections: {images_with_no_detection}",
        f"Avg inference time (ms) : {avg_time:.2f}",
        f"Avg FPS                 : {avg_fps:.2f}",
        "",
        "Detections per class:",
    ]
    for cls_name, count in sorted(class_counter.items(), key=lambda x: -x[1]):
        summary_lines.append(f"  {cls_name:<20}: {count}")

    summary_lines += [
        "",
        "Overall validation metrics (from --data, if provided):",
        f"  Precision   : {overall_metrics['precision']}",
        f"  Recall      : {overall_metrics['recall']}",
        f"  mAP@0.5     : {overall_metrics['map50']}",
        f"  mAP@0.5:0.95: {overall_metrics['map5095']}",
        "",
        "Per-class validation metrics:",
    ]
    for cls_name, m in per_class_metrics.items():
        summary_lines.append(
            f"  {cls_name:<20}: P={m['precision']}  R={m['recall']}  mAP50={m['map50']}  mAP50-95={m['map5095']}"
        )

    summary_path = out_root / "summary.txt"
    with open(summary_path, "w") as f:
        f.write("\n".join(summary_lines) + "\n")

    logger.info(f"Saved {summary_path}")
    logger.info(f"Annotated images saved to {img_out_dir}")
    logger.info("=== Inference run complete ===")


if __name__ == "__main__":
    main()
