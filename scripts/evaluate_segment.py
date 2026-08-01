import argparse
import csv
import logging
import logging.handlers
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np
import cv2

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import yaml
except ImportError:
    yaml = None

MASK_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def parse_args():
    p = argparse.ArgumentParser(description="Full IoU/Dice/PixelAcc/Precision/Recall evaluation from semantic masks")
    p.add_argument("--pred-dirs", nargs="+", required=True, help="Predicted masks_semantic dir, one per model tag")
    p.add_argument("--gt-dirs", nargs="+", required=True, help="GT masks_semantic dir, one per model tag")
    p.add_argument("--tags", nargs="+", required=True, help="Model tags, e.g. yolov11_n yolov11_s yolov11_m")
    p.add_argument("--data", default=None, help="Path to data.yaml (reads class names from 'names')")
    p.add_argument("--class-names", default=None, help="Comma-separated class names, used if --data not given")
    p.add_argument("--output", default="Output/Evaluation_Full", help="Root output directory")
    p.add_argument("--devices", default="cuda:0,cuda:1,cuda:2",
                    help="Comma-separated devices, one per tag (cycled if fewer). Use 'cpu' to force CPU.")
    return p.parse_args()


def load_class_names(args) -> list:
    if args.data:
        if yaml is None:
            sys.exit("pyyaml not installed but --data was given. pip install pyyaml")
        with open(args.data, "r") as f:
            d = yaml.safe_load(f)
        names = d.get("names")
        if isinstance(names, dict):
            names = [names[k] for k in sorted(names.keys(), key=lambda x: int(x))]
        return list(names)
    if args.class_names:
        return [n.strip() for n in args.class_names.split(",")]
    sys.exit("Provide either --data (data.yaml) or --class-names (comma-separated).")


def worker_configurer(log_queue: mp.Queue):
    h = logging.handlers.QueueHandler(log_queue)
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(h)
    root.setLevel(logging.INFO)


def safe_div(a, b):
    return (a / b) if b > 0 else float("nan")


def evaluate_one_model(pred_dir, gt_dir, tag, device_str, num_classes, class_names, out_root, log_queue):
    worker_configurer(log_queue)
    logger = logging.getLogger(tag)
    t_start = time.time()

    use_torch = TORCH_AVAILABLE and device_str != "cpu"
    if use_torch:
        try:
            device = torch.device(device_str)
            torch.zeros(1, device=device)
        except Exception as e:
            logger.warning(f"[{tag}] requested device '{device_str}' unusable ({e}); falling back to CPU.")
            use_torch = False
            device_str = "cpu"

    logger.info(f"[{tag}] starting. pred={pred_dir}  gt={gt_dir}  device={device_str}")

    pred_dir, gt_dir = Path(pred_dir), Path(gt_dir)
    gt_files = sorted([p for p in gt_dir.iterdir() if p.suffix.lower() in MASK_EXTS])
    if not gt_files:
        logger.error(f"[{tag}] no GT mask files found in {gt_dir}")
        return {"tag": tag, "miou": float("nan"), "mdice": float("nan"), "mprec": float("nan"),
                "mrec": float("nan"), "pixel_acc": float("nan"), "rows": [], "n_eval": 0, "n_missing": 0}

    inter = np.zeros(num_classes + 1, dtype=np.int64)      # TP per class
    pred_sum = np.zeros(num_classes + 1, dtype=np.int64)   # TP+FP per class
    gt_sum = np.zeros(num_classes + 1, dtype=np.int64)     # TP+FN per class
    global_correct = 0
    global_total = 0
    n_eval, n_missing = 0, 0

    for gt_path in gt_files:
        pred_path = pred_dir / gt_path.name
        if not pred_path.exists():
            n_missing += 1
            logger.warning(f"[{tag}] missing prediction for {gt_path.name}, skipped")
            continue

        gt_mask = cv2.imread(str(gt_path), cv2.IMREAD_UNCHANGED)
        pred_mask = cv2.imread(str(pred_path), cv2.IMREAD_UNCHANGED)
        if gt_mask is None or pred_mask is None:
            n_missing += 1
            logger.warning(f"[{tag}] could not read mask pair for {gt_path.name}, skipped")
            continue
        if pred_mask.shape != gt_mask.shape:
            pred_mask = cv2.resize(pred_mask, (gt_mask.shape[1], gt_mask.shape[0]), interpolation=cv2.INTER_NEAREST)
            logger.warning(f"[{tag}] resized prediction to match GT shape for {gt_path.name}")

        h, w = gt_mask.shape[:2]
        global_total += h * w

        if use_torch:
            g = torch.from_numpy(gt_mask.astype(np.int32)).to(device)
            p = torch.from_numpy(pred_mask.astype(np.int32)).to(device)
            global_correct += int((g == p).sum().item())
            for c in range(1, num_classes + 1):
                gc, pc = (g == c), (p == c)
                inter[c] += int((gc & pc).sum().item())
                pred_sum[c] += int(pc.sum().item())
                gt_sum[c] += int(gc.sum().item())
        else:
            g, p = gt_mask.astype(np.int32), pred_mask.astype(np.int32)
            global_correct += int((g == p).sum())
            for c in range(1, num_classes + 1):
                gc, pc = (g == c), (p == c)
                inter[c] += int(np.logical_and(gc, pc).sum())
                pred_sum[c] += int(pc.sum())
                gt_sum[c] += int(gc.sum())

        n_eval += 1
        if n_eval % 25 == 0:
            logger.info(f"[{tag}] processed {n_eval}/{len(gt_files)} images")

    rows = []
    iou_list, dice_list, prec_list, rec_list = [], [], [], []
    for c in range(1, num_classes + 1):
        name = class_names[c - 1] if c - 1 < len(class_names) else f"class_{c}"
        tp = inter[c]
        fp = pred_sum[c] - tp
        fn = gt_sum[c] - tp
        tn = global_total - tp - fp - fn

        union = tp + fp + fn
        iou = safe_div(tp, union)
        dice = safe_div(2 * tp, 2 * tp + fp + fn)
        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        accuracy = safe_div(tp + tn, global_total) if global_total > 0 else float("nan")

        if not np.isnan(iou):
            iou_list.append(iou)
        if not np.isnan(dice):
            dice_list.append(dice)
        if not np.isnan(precision):
            prec_list.append(precision)
        if not np.isnan(recall):
            rec_list.append(recall)

        rows.append([
            c - 1, name, int(tp), int(fp), int(fn), int(tn),
            round(float(iou), 4) if not np.isnan(iou) else "N/A",
            round(float(dice), 4) if not np.isnan(dice) else "N/A",
            round(float(precision), 4) if not np.isnan(precision) else "N/A",
            round(float(recall), 4) if not np.isnan(recall) else "N/A",
            round(float(accuracy), 4) if not np.isnan(accuracy) else "N/A",
        ])

    miou = float(np.mean(iou_list)) if iou_list else float("nan")
    mdice = float(np.mean(dice_list)) if dice_list else float("nan")
    mprec = float(np.mean(prec_list)) if prec_list else float("nan")
    mrec = float(np.mean(rec_list)) if rec_list else float("nan")
    pixel_acc = safe_div(global_correct, global_total)

    out_dir = Path(out_root) / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "class_metrics.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class_id", "class_name", "TP", "FP", "FN", "TN",
                    "iou", "dice", "precision", "recall", "accuracy"])
        w.writerows(rows)

    elapsed = time.time() - t_start
    summary_lines = [
        f"Full Segmentation Evaluation Summary — {tag}",
        "=" * 60,
        f"Predicted masks : {pred_dir}",
        f"GT masks        : {gt_dir}",
        f"Device used     : {device_str if use_torch else 'cpu'}",
        f"Images evaluated: {n_eval}",
        f"Images missing/skipped: {n_missing}",
        f"Eval time (s)   : {elapsed:.2f}",
        "",
        f"mIoU               : {round(miou, 4) if not np.isnan(miou) else 'N/A'}",
        f"mDice              : {round(mdice, 4) if not np.isnan(mdice) else 'N/A'}",
        f"Mean Precision     : {round(mprec, 4) if not np.isnan(mprec) else 'N/A'}",
        f"Mean Recall        : {round(mrec, 4) if not np.isnan(mrec) else 'N/A'}",
        f"Overall Pixel Acc. : {round(pixel_acc, 4) if not np.isnan(pixel_acc) else 'N/A'}  "
        f"(all classes + background, dataset-wide)",
        "",
        "Per-class (IoU / Dice / Precision / Recall / Accuracy):",
    ]
    for r in rows:
        summary_lines.append(
            f"  {r[1]:<20}: IoU={r[6]}  Dice={r[7]}  P={r[8]}  R={r[9]}  Acc={r[10]}"
        )
    (out_dir / "summary.txt").write_text("\n".join(summary_lines) + "\n")

    logger.info(f"[{tag}] DONE in {elapsed:.2f}s. mIoU={miou:.4f} mDice={mdice:.4f} "
                f"mPrec={mprec:.4f} mRec={mrec:.4f} PixelAcc={pixel_acc:.4f} "
                f"(evaluated {n_eval}, missing {n_missing})")

    return {"tag": tag, "miou": miou, "mdice": mdice, "mprec": mprec, "mrec": mrec,
            "pixel_acc": pixel_acc, "rows": rows, "n_eval": n_eval, "n_missing": n_missing}


def main():
    args = parse_args()
    if not (len(args.pred_dirs) == len(args.gt_dirs) == len(args.tags)):
        sys.exit("--pred-dirs, --gt-dirs and --tags must all have the same number of entries.")

    class_names = load_class_names(args)
    num_classes = len(class_names)

    out_root = Path(args.output)
    out_root.mkdir(parents=True, exist_ok=True)
    log_path = out_root / "evaluation.log"

    mp_ctx = mp.get_context("spawn")
    manager = mp_ctx.Manager()
    log_queue = manager.Queue()

    fh = logging.FileHandler(log_path, mode="w")
    fh.setFormatter(logging.Formatter("%(asctime)s | %(processName)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S"))
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(logging.Formatter("%(processName)s | %(message)s"))
    listener = logging.handlers.QueueListener(log_queue, fh, sh)
    listener.start()

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(logging.handlers.QueueHandler(log_queue))
    root_logger.setLevel(logging.INFO)

    devices = [d.strip() for d in args.devices.split(",")] if args.devices else ["cpu"] * len(args.tags)
    if len(devices) < len(args.tags):
        devices = (devices * len(args.tags))[: len(args.tags)]

    if not TORCH_AVAILABLE:
        logging.warning("torch not available in this environment; all models will run on CPU (numpy).")
        devices = ["cpu"] * len(args.tags)

    logging.info(f"Evaluating {len(args.tags)} model(s): {args.tags}")
    logging.info(f"Class names ({num_classes}): {class_names}")
    logging.info(f"Devices assigned: {dict(zip(args.tags, devices))}")

    t0 = time.time()
    with mp_ctx.Pool(processes=len(args.tags)) as pool:
        async_results = [
            pool.apply_async(
                evaluate_one_model,
                (pred_dir, gt_dir, tag, device, num_classes, class_names, str(out_root), log_queue),
            )
            for pred_dir, gt_dir, tag, device in zip(args.pred_dirs, args.gt_dirs, args.tags, devices)
        ]
        results = [r.get() for r in async_results]

    listener.stop()
    total_elapsed = time.time() - t0

    with open(out_root / "comparison_summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tag", "mIoU", "mDice", "mean_precision", "mean_recall",
                    "overall_pixel_accuracy", "images_evaluated", "images_missing"])
        for r in results:
            w.writerow([
                r["tag"],
                round(r["miou"], 4) if not np.isnan(r["miou"]) else "N/A",
                round(r["mdice"], 4) if not np.isnan(r["mdice"]) else "N/A",
                round(r["mprec"], 4) if not np.isnan(r["mprec"]) else "N/A",
                round(r["mrec"], 4) if not np.isnan(r["mrec"]) else "N/A",
                round(r["pixel_acc"], 4) if not np.isnan(r["pixel_acc"]) else "N/A",
                r["n_eval"], r["n_missing"],
            ])

    # class x model matrix: iou, dice, precision, recall per model
    per_tag_rows = {r["tag"]: {row[1]: (row[6], row[7], row[8], row[9]) for row in r["rows"]} for r in results}
    with open(out_root / "comparison_full.csv", "w", newline="") as f:
        w = csv.writer(f)
        header = ["class_id", "class_name"]
        for r in results:
            header += [f"{r['tag']}_iou", f"{r['tag']}_dice", f"{r['tag']}_precision", f"{r['tag']}_recall"]
        w.writerow(header)
        for c, cname in enumerate(class_names):
            row = [c, cname]
            for r in results:
                iou, dice, prec, rec = per_tag_rows.get(r["tag"], {}).get(cname, ("N/A", "N/A", "N/A", "N/A"))
                row += [iou, dice, prec, rec]
            w.writerow(row)

    best_by_miou = max(results, key=lambda r: (r["miou"] if not np.isnan(r["miou"]) else -1))
    lines = [
        "Model Comparison Summary — Full Metrics (SAM2 vs SAM-HQ pseudo-GT)",
        "=" * 70,
        f"Total wall-clock time (all models, parallel): {total_elapsed:.2f}s",
        "",
    ]
    for r in results:
        lines.append(
            f"  {r['tag']:<15}: mIoU={round(r['miou'],4) if not np.isnan(r['miou']) else 'N/A'}   "
            f"mDice={round(r['mdice'],4) if not np.isnan(r['mdice']) else 'N/A'}   "
            f"mPrec={round(r['mprec'],4) if not np.isnan(r['mprec']) else 'N/A'}   "
            f"mRec={round(r['mrec'],4) if not np.isnan(r['mrec']) else 'N/A'}   "
            f"PixelAcc={round(r['pixel_acc'],4) if not np.isnan(r['pixel_acc']) else 'N/A'}   "
            f"(evaluated {r['n_eval']}, missing {r['n_missing']})"
        )
    lines += ["", f"Best overall (by mIoU): {best_by_miou['tag']}", ""]

    if "mine" in class_names:
        lines.append("Mine-class comparison:")
        for r in results:
            iou, dice, prec, rec = per_tag_rows.get(r["tag"], {}).get("mine", ("N/A", "N/A", "N/A", "N/A"))
            lines.append(f"  {r['tag']:<15}: IoU={iou}  Dice={dice}  Precision={prec}  Recall={rec}")

    (out_root / "comparison_summary.txt").write_text("\n".join(lines) + "\n")

    logging.info(f"All models done in {total_elapsed:.2f}s (wall-clock, parallel across devices).")
    logging.info(f"Saved: {out_root / 'comparison_summary.csv'}, {out_root / 'comparison_full.csv'}, "
                 f"{out_root / 'comparison_summary.txt'}")
    print(f"\nDone. Results saved under: {out_root}")


if __name__ == "__main__":
    main()
