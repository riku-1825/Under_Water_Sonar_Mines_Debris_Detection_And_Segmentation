#!/usr/bin/env python3
import argparse
import logging
import os
import time
import numpy as np
from pathlib import Path

import cv2
from sam2.build_sam import build_sam2
import torch
from ultralytics import YOLO


def setup_logger(out_dir):
    os.makedirs(out_dir, exist_ok=True)

    logger = logging.getLogger("YOLO_SAM2")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    fh = logging.FileHandler(os.path.join(out_dir, "inference.log"))
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    return logger


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--images", required=True)
    p.add_argument("--yolo", required=True)
    p.add_argument("--sam-config", required=True)
    p.add_argument("--sam-checkpoint", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument(
        "--min-contour-area",
        type=float,
        default=10.0,
        help="Discard tiny contour fragments (in pixels) when converting masks to polygons.",
    )
    p.add_argument(
        "--poly-epsilon",
        type=float,
        default=0.002,
        help="approxPolyDP epsilon as a fraction of the contour perimeter (smaller = more points).",
    )
    return p.parse_args()


def colour_mask(mask, colour=(0, 0, 255)):
    img = np.zeros((*mask.shape, 3), dtype=np.uint8)
    img[mask.astype(bool)] = colour
    return img


def overlay(image, mask, colour=(0, 0, 255), alpha=0.45):
    out = image.copy()
    c = np.zeros_like(image)
    c[mask.astype(bool)] = colour
    return cv2.addWeighted(out, 1.0, c, alpha, 0)


def get_palette(num_classes):
    """Deterministic BGR palette. Index 0 is reserved for background (black)."""
    rng = np.random.RandomState(42)
    palette = [(0, 0, 0)]
    hues = np.linspace(0, 179, num_classes, endpoint=False)
    for h in hues:
        hsv = np.uint8([[[h, 200, 255]]])
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0].tolist()
        palette.append(tuple(int(v) for v in bgr))
    return palette


def colorize_semantic(semantic_mask, palette):
    h, w = semantic_mask.shape
    color_img = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id in np.unique(semantic_mask):
        if class_id == 0:
            continue
        color = palette[int(class_id) % len(palette)]
        color_img[semantic_mask == class_id] = color
    return color_img


def mask_to_yolo_polygons(mask, class_id, img_w, img_h, min_area=10.0, epsilon_frac=0.002):
    """Convert a binary instance mask into one or more YOLO-seg label lines:
    'class_id x1 y1 x2 y2 ... xn yn' with coordinates normalized to [0, 1].
    A mask can produce more than one polygon if it has disjoint regions.
    """
    contours, _ = cv2.findContours(
        (mask.astype(np.uint8)) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    lines = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue

        perimeter = cv2.arcLength(contour, True)
        epsilon = epsilon_frac * perimeter
        approx = cv2.approxPolyDP(contour, epsilon, True)

        if len(approx) < 3:
            continue

        coords = approx.reshape(-1, 2).astype(np.float64)
        coords[:, 0] /= img_w
        coords[:, 1] /= img_h
        coords = np.clip(coords, 0.0, 1.0)

        flat = " ".join(f"{v:.6f}" for v in coords.flatten())
        lines.append(f"{class_id} {flat}")

    return lines


def main():
    args = parse_args()

    out = Path(args.output)
    (out / "masks").mkdir(parents=True, exist_ok=True)
    (out / "overlays").mkdir(exist_ok=True)
    (out / "json").mkdir(exist_ok=True)
    (out / "semantic").mkdir(exist_ok=True)
    (out / "semantic_color").mkdir(exist_ok=True)
    (out / "labels").mkdir(exist_ok=True)

    logger = setup_logger(args.output)

    logger.info("=" * 80)
    logger.info("YOLO + SAM2 Inference Started")
    logger.info("Images         : %s", args.images)
    logger.info("YOLO Weights   : %s", args.yolo)
    logger.info("SAM Config     : %s", args.sam_config)
    logger.info("SAM Checkpoint : %s", args.sam_checkpoint)
    logger.info("Device         : %s", args.device)

    if torch.cuda.is_available():
        logger.info("GPU            : %s", torch.cuda.get_device_name(0))
        logger.info("CUDA Version   : %s", torch.version.cuda)

    # -------- Load YOLO --------
    yolo = YOLO(args.yolo)

    # class-id -> class-name legend, and a stable colour palette for the
    # semantic masks (index 0 is background / "no detection")
    class_names = yolo.names  # dict: {id: name}
    num_classes = max(class_names.keys()) + 1
    palette = get_palette(num_classes)

    with open(out / "classes.txt", "w") as f:
        f.write("0: background\n")
        for cid in sorted(class_names.keys()):
            f.write(f"{cid + 1}: {class_names[cid]}\n")

    # -------- Load SAM2 --------
    from hydra import initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

    initialize_config_dir(
        config_dir="/mnt/DATA/EE25M305/Sonar_Vision/sam2/sam2/configs",
        version_base="1.2",
    )

    logger.info("Loading SAM2 model...")

    sam_model = build_sam2(
        config_file=args.sam_config,
        ckpt_path=args.sam_checkpoint,
        device=args.device,
    )

    predictor = SAM2ImagePredictor(sam_model)

    logger.info("SAM2 model loaded successfully.")

    image_dir = Path(args.images)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    images = sorted([p for p in image_dir.iterdir() if p.suffix.lower() in exts])

    logger.info("Total Images : %d", len(images))

    total_time = 0.0

    for idx, img_path in enumerate(images, 1):
        start = time.time()

        img = cv2.imread(str(img_path))
        if img is None:
            logger.warning("Cannot read %s", img_path)
            continue

        _ = yolo.predict(
            source=img,
            conf=args.conf,
            device=args.device,
            verbose=False,
        )

        # ---- SAM2 inference goes here ----

        elapsed = time.time() - start
        total_time += elapsed

        logger.info(
            "[%d/%d] %s | Time: %.4f sec | FPS: %.2f",
            idx,
            len(images),
            img_path.name,
            elapsed,
            1.0 / elapsed if elapsed > 0 else 0,
        )

    logger.info("-" * 80)
    logger.info("Inference Finished")
    logger.info("Processed Images : %d", len(images))
    logger.info("Total Time       : %.2f sec", total_time)
    logger.info("Average Time     : %.4f sec/image",
                total_time / max(len(images), 1))
    logger.info("Average FPS      : %.2f",
                len(images) / total_time if total_time > 0 else 0)

    if torch.cuda.is_available():
        logger.info(
            "Peak GPU Memory : %.2f GB",
            torch.cuda.max_memory_allocated() / (1024 ** 3),
        )

    # ===================== Detection + Segmentation =====================
    import json
    import pandas as pd
    from tqdm import tqdm

    IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

    rows = []

    images = sorted([p for p in Path(args.images).rglob("*") if p.suffix.lower() in IMG_EXT])

    for img_path in tqdm(images):
        image = cv2.imread(str(img_path))
        if image is None:
            continue

        img_h, img_w = image.shape[:2]

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        predictor.set_image(rgb)

        result = yolo.predict(
            source=image,
            conf=args.conf,
            verbose=False,
            device=args.device
        )[0]

        json_items = []

        semantic_mask = np.zeros((img_h, img_w), dtype=np.uint8)
        label_lines = []

        if result.boxes is None or len(result.boxes) == 0:
            cv2.imwrite(str(out / "semantic" / f"{img_path.stem}.png"), semantic_mask)
            cv2.imwrite(
                str(out / "semantic_color" / img_path.name),
                colorize_semantic(semantic_mask, palette),
            )
            with open(out / "labels" / f"{img_path.stem}.txt", "w") as f:
                pass
            continue

        boxes = result.boxes.xyxy.cpu().numpy()
        cls = result.boxes.cls.cpu().numpy().astype(int)
        confs = result.boxes.conf.cpu().numpy()

        combined = image.copy()

        for idx, (box, c, s) in enumerate(zip(boxes, cls, confs)):
            masks, scores, _ = predictor.predict(
                box=box,
                multimask_output=False
            )

            mask = masks[0].astype(np.uint8)
            sam_score = float(scores[0])

            mask_name = f"{img_path.stem}_{idx}.png"
            cv2.imwrite(str(out / "masks" / mask_name), mask * 255)

            # Stamp this instance's class id (offset by +1, since 0 is
            # background) into the shared semantic mask for the image.
            # Later instances win where masks overlap.
            semantic_mask[mask.astype(bool)] = int(c) + 1

            label_lines.extend(
                mask_to_yolo_polygons(
                    mask,
                    class_id=int(c),
                    img_w=img_w,
                    img_h=img_h,
                    min_area=args.min_contour_area,
                    epsilon_frac=args.poly_epsilon,
                )
            )

            combined = overlay(combined, mask)

            x1, y1, x2, y2 = map(int, box)

            cv2.rectangle(combined, (x1, y1), (x2, y2), (0, 255, 0), 2)

            cv2.putText(
                combined,
                f"{c}:{s:.2f}",
                (x1, max(15, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1
            )

            rows.append({
                "image": img_path.name,
                "object": idx,
                "class": int(c),
                "yolo_conf": float(s),
                "sam_score": sam_score,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2
            })

            json_items.append({
                "bbox": [x1, y1, x2, y2],
                "class": int(c),
                "yolo_conf": float(s),
                "sam_score": sam_score,
                "mask": mask_name
            })

        cv2.imwrite(str(out / "overlays" / img_path.name), combined)

        with open(out / "json" / f"{img_path.stem}.json", "w") as f:
            json.dump(json_items, f, indent=2)

        # ---- semantic mask + YOLO-seg label outputs for this image ----
        
        #cv2.imwrite(str(out / "masks_semantic" / f"{img_path.stem}.png"), semantic_mask)
        
        cv2.imwrite(str(out / "semantic" / img_path.name),semantic_mask)
        cv2.imwrite(
            str(out / "semantic_color" / img_path.name),
            colorize_semantic(semantic_mask, palette),
        )
        with open(out / "labels" / f"{img_path.stem}.txt", "w") as f:
            f.write("\n".join(label_lines))

    pd.DataFrame(rows).to_csv(out / "detections.csv", index=False)
    # ============================================================

    logger.info("=" * 80)
    print("Inference completed. Check the log file for details.")


if __name__ == "__main__":
    main()
