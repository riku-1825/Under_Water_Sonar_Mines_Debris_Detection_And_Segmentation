![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)
![CUDA](https://img.shields.io/badge/CUDA-Enabled-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

## Overview

The goal is to study how well modern vision models perform on underwater sonar imagery, where low contrast, noise, shadowing, and object ambiguity make detection and segmentation difficult.

This repository contains my work on underwater sonar-based **object detection** and **segmentation** for mines and debris. The project compares multiple **detection** algorithms like **YOLOV11n**, **YOLOV11s** & **YOLOV11m** and **segmentation** approaches with **SAM2** for segmentation on forward-looking sonar images. As the ground truth mask of sonar mine images were not available, pseudo-masking is done with the help of **SAM-HQ** and considered it as ground truth masking for further evaluation.

For **SAM2**; **sam2.1_hiera_small.pt** has been used and for **SAMHQ**; **sam_hq_vit_l.pth** has been used. Onedrive link is provided below.

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
  <a href="https://1drv.ms/f/c/279f2820d299d227/IgCF2mbMuF-hQYbhLFr7qf5oAS8bzLrGDYSzmfX1-1Rf9xM?e=OPzcTh">
    <img src="https://img.shields.io/badge/Download-Model%20Checkpoints-0078D4?style=for-the-badge&logo=microsoftonedrive&logoColor=white">
  </a>
</p>

The download package includes:

- **SAM2**
  - sam2.1_hiera_small.pt

- **SAM-HQ**
  - sam_hq_vit_l.pth

---

## How to Run

The table below summarizes the purpose of each script in this repository.

| Task | Script | Description |
|------|--------|-------------|
| Training (YOLOv11n) | `scripts/train_yolov11_n.py` | Performs training on train data using the YOLOv11n model. |
| Training (YOLOv11s) | `scripts/train_yolov11_s.py` | Performs training on train data using the YOLOv11s model. |
| Training (YOLOv11m) | `scripts/train_yolov11_m.py` | Performs training on train data using the YOLOv11m model. |
| Segmentation using Sam2 | `scripts/segment.sh` | Performs segmentation using Sam2 on test data. |
| Producing pseudo-masking using sam-hq | `scripts/run_generate_gt.sh` | Performs segmentation to generate pseudo-masking on test data. |
| Detection on test data | `scripts/detect.sh` | Performs detection task on test data and does evaluation. |
| Evaluation for segmentation | `scripts/evaluate_segment` | Performs evaluation for segmentation task. |

---

---

## Detection Result Comparison (Overall)

| Model | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 | Avg. Inference Time (ms) | Avg. FPS |
|:------:|:---------:|:------:|:--------:|:------------:|:-------------------:|:---:|
| YOLOv11n | **0.9730** | 0.9831 | **0.9889** | **0.7735** | **13.72** | **76.18** |
| YOLOv11s | 0.9766 | **0.9896** | 0.9830 | 0.7630 | 14.38 | 72.94 |
| YOLOv11m | **0.9828** | 0.9688 | 0.9795 | 0.7570 | 16.20 | 64.16 |

## Detection Result Comparison (Mine Class)

| Model | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 | Avg. Inference Time (ms) | Avg. FPS |
|:------:|:---------:|:------:|:--------:|:------------:|:------------------------:|:--------:|
| YOLOv11n | **0.9492** | 0.9344 | **0.9585** | **0.5123** | **13.22** | **75.94** |
| YOLOv11s | 0.9474 | **0.9500** | 0.9075 | 0.4551 | 13.43 | 74.99 |
| YOLOv11m | 0.9435 | **0.9500** | 0.8978 | 0.3824 | 15.10 | 66.54 |

### Key Observations

- **YOLOv11n** achieved the best overall performance, delivering the highest mAP@0.5:0.95 while maintaining the fastest inference speed (76.18 FPS).
- For the **Mine** class, **YOLOv11n** obtained the highest precision and mAP, whereas **YOLOv11s** achieved the highest recall (95.00%), indicating better sensitivity to mine detection.
- **YOLOv11m** required the longest inference time and achieved the lowest FPS, while offering no significant improvement in detection accuracy over the smaller YOLOv11 variants.

---
---
## Segmentation Result Comparison (Overall)

| Model | mIoU | mDice | Precision | Recall | Pixel Accuracy | Evaluation Time (s) |
|:------:|:----:|:-----:|:---------:|:------:|:--------------:|:-------------------:|
| YOLOv11n + SAM2 | 0.5044 | 0.6352 | 0.5654 | 0.7606 | 0.9904 | 3.97 |
| YOLOv11s + SAM2 | 0.4955 | 0.6312 | 0.5535 | **0.7720** | 0.9903 | 3.90 |
| YOLOv11m + SAM2 | **0.5066** | **0.6401** | **0.5676** | 0.7649 | **0.9908** | **3.71** |

## Segmentation Result Comparison (Mine Class)

| Model | IoU | Dice | Precision | Recall |
|:------:|:---:|:----:|:---------:|:------:|
| YOLOv11n + SAM2 | **0.5785** | **0.7329** | 0.5789 | 0.9986 |
| YOLOv11s + SAM2 | 0.5112 | 0.6765 | 0.5114 | **0.9990** |
| YOLOv11m + SAM2 | 0.5746 | 0.7299 | **0.5815** | 0.9800 |

### Key Observations

- **YOLOv11m + SAM2** achieved the best overall segmentation performance, obtaining the highest mIoU (0.5066), Dice score (0.6401), Precision (0.5676), and Pixel Accuracy (0.9908), while also having the shortest evaluation time.
- For the **Mine** class, **YOLOv11n + SAM2** produced the highest IoU (0.5785) and Dice score (0.7329), whereas **YOLOv11m + SAM2** achieved the highest precision (0.5815).
- All three models achieved **very high recall (≥ 0.98)** for the Mine class, indicating that the segmentation pipeline successfully captured nearly all mine regions with only minor differences in boundary accuracy.

---

## References

This project builds upon the following open-source repositories:

### YOLOv11

- **Ultralytics YOLO**  
  GitHub: https://github.com/ultralytics/ultralytics

  ```bibtex
  @software{Jocher_YOLO,
    author = {Glenn Jocher and Jing Qiu},
    title = {Ultralytics YOLO},
    year = {2025},
    publisher = {GitHub},
    url = {https://github.com/ultralytics/ultralytics}
  }
  ```
### SAM2 (Segment Anything Model 2)

- **Official GitHub Repository**  
  https://github.com/facebookresearch/sam2

- **Paper**  
  Ravi, N., Gabeur, V., Hu, Y.-T., et al. (2024). *SAM 2: Segment Anything in Images and Videos*. arXiv:2408.00714.

  ```bibtex
  @article{ravi2024sam2,
    title={SAM 2: Segment Anything in Images and Videos},
    author={Ravi, Nikhila and Gabeur, Valentin and Hu, Yuan-Ting and others},
    journal={arXiv preprint arXiv:2408.00714},
    year={2024}
  }
  ```
### SAM-HQ

- **Official GitHub Repository**  
  https://github.com/SysCV/sam-hq

- **Paper**  
  Ke, L., Ye, M., Danelljan, M., Tai, Y.-W., Tang, C.-K., Yu, F. (2023). *Segment Anything in High Quality*. NeurIPS 2023.

  ```bibtex
  @inproceedings{ke2023samhq,
    title={Segment Anything in High Quality},
    author={Ke, Lei and Ye, Mingqiang and Danelljan, Martin and Tai, Yu-Wing and Tang, Chi-Keung and Yu, Fisher},
    booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
    year={2023}
  }
  ```

---

## Citation

If you find this repository useful in your research, please consider citing it:

```bibtex
@misc{bagh2026underwatersonar,
  author       = {Bhoumik Chandra Bagh},
  title        = {Under Water Sonar Mines and Debris Detection and Segmentation},
  year         = {2026},
  publisher    = {GitHub},
  howpublished = {\url{https://github.com/riku-1825/Under_Water_Sonar_Mines_Debris_Detection_And_Segmentation}},
  note         = {GitHub repository}
}
```

---

## License

This project is licensed under the **MIT License**.

You are free to:

- Use the source code for research and educational purposes.
- Modify and distribute the code under the terms of the MIT License.
- Include this work in your own projects with appropriate attribution.

For more details, see the [LICENSE](LICENSE) file.

---

