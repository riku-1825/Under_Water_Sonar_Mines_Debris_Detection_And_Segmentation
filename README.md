![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)
![CUDA](https://img.shields.io/badge/CUDA-Enabled-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

## Overview

The goal is to study how well modern vision models perform on underwater sonar imagery, where low contrast, noise, shadowing, and object ambiguity make detection and segmentation difficult.

This repository contains my work on underwater sonar-based **object detection** and **segmentation** for mines and debris. The project compares multiple **detection** algorithms like **YOLOV11n**, **YOLOV11s** & **YOLOV11m** and **segmentation** approaches with **SAM2** for segmentation on forward-looking sonar images. As the ground truth mask of sonar mine images were not available, pseudo-masking is done with the help of **SAM-HQ** and considered it as ground truth masking for further evaluation.

For **SAM2**; *sam2.1_hiera_small.pt* has been used and for **SAMHQ**; *sam_hq_vit_l.pth*. Onedrive link is provided below.
---

## Sample Results

<table align="center">
  <tr>
    <td align="center">
      <img src="https://github.com/riku-1825/Under_Water_Sonar_Mines_Debris_Detection_And_Segmentation/blob/main/Sonar_Dataset/images/test/mine_sonar86_jpg.rf.a7bc6bc21f73e1636afa72a097c8878d.jpg" width="320"><br>
      <b>Normal</b>
    </td>
    <td align="center">
      <img src="https://github.com/riku-1825/Under_Water_Sonar_Mines_Debris_Detection_And_Segmentation/blob/main/Output/Detection/yolov11_n/images/mine_sonar86_jpg.rf.a7bc6bc21f73e1636afa72a097c8878d.jpg" width="320"><br>
      <b>Detected</b>
    </td>
    <td align="center">
      <img src="https://github.com/riku-1825/Under_Water_Sonar_Mines_Debris_Detection_And_Segmentation/blob/main/Output/Segment_sam2/yolov11_n/overlays/mine_sonar86_jpg.rf.a7bc6bc21f73e1636afa72a097c8878d.jpg" width="320"><br>
      <b>Segmented</b>
    </td>
  </tr>
</table>

---

---

## Features

- Underwater sonar object **detection**
- Underwater sonar object **segmentation**
- Comparison of **YOLOv11n / YOLOv11s / YOLOv11m**
- performance of **SAM2**
- End-to-end pipeline for inference and result generation
- Visual output for qualitative comparison
- Supports mines and debris sonar imagery

---

## Repository Structure

```text

├── Output/                   # Detection, Segmentation, Evaluation Output
├── Sonar_Dataset             # Dataset with images & lebels
├── run/train                 # Best trained weights for yolov11 versions
├── scripts/                  # Scripts for training / inference / evaluation
├── LICENSE
├── README.md

```
---

---

## Dataset

This project uses a publicly available **Forward-Looking Sonar (FLS)** dataset for underwater object detection and segmentation. The dataset contains sonar images of underwater **mines** and **marine debris**, along with object detection annotations.

### Dataset Links

| Source | Link |
|:------:|------|
| GitHub | https://github.com/riku-1825/Forward_Looking_Under_Water_Mines_And_Debris_Sonar_dataset |
| Kaggle | https://www.kaggle.com/datasets/bhoumikchandrabagh/forward-looking-sonar-object-detection-dataset |

### Dataset Statistics

| Property | Value |
|----------|-------|
| Image Type | Forward-Looking Sonar (FLS) |
| Classes | Mine, Can, Bottle, Chain, Drink Carton, Hook, Propeller, Shampoo Bottle, Standing Bottle, Tire, Valve |
| Number of Classes | 11 |
| Annotation Format | YOLO |
| Task | Object Detection & Segmentation |

---

---
## Installing SAM2 and SAM-HQ

Clone the official repositories:

### SAM2

```bash
git clone https://github.com/facebookresearch/sam2.git
cd sam2

pip install -e .
```
---
### SAM-HQ

```bash
git clone https://github.com/SysCV/sam-hq.git
cd sam-hq

pip install -e .
```
---

---
## Model Checkpoints

The pretrained checkpoints required for segmentation are available from the OneDrive link below.

<p align="center">
  <a href="https://YOUR_ONEDRIVE_LINK_HERE">
    <img src="https://img.shields.io/badge/Download-Model%20Checkpoints-0078D4?style=for-the-badge&logo=microsoftonedrive&logoColor=white">
  </a>
</p>

The download package includes:

- **SAM2**
  - sam2.1_hiera_small.pt

- **SAM-HQ**
  - sam_hq_vit_l.pth

> **Note:** Download the checkpoints and place them inside the `sam2/checkpoints/` & `sam-hq/checkpoints/` directory before running the segmentation scripts.

---

