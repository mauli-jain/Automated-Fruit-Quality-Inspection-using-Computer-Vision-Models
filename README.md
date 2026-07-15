# Automated Fruit Quality Inspection using Computer Vision

![Python](https://img.shields.io/badge/Python-3.x-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![Ultralytics](https://img.shields.io/badge/YOLOv8--Seg-Ultralytics-purple)
![HuggingFace](https://img.shields.io/badge/🤗-Transformers-yellow)
![Status](https://img.shields.io/badge/status-completed-brightgreen)
![License](https://img.shields.io/badge/license-MIT-green)

A four-phase deep learning pipeline for automated fruit quality inspection — progressing from image classification to transformer-based semantic segmentation — built during a **Practice School-I (PS-1) at UST – Infinity Labs, Thiruvananthapuram**, in fulfillment of BITS Pilani, Hyderabad Campus requirements (25 May – 18 Jul 2026).

📄 **Full report (44 pages, with all figures/appendices):** [`Report/UST_REPORT_MAULI_JAIN_311410.pdf`](Report/UST_REPORT_MAULI_JAIN_311410.pdf )

---

## 📑 Table of Contents

- [Project Overview](#-project-overview)
- [Research Motivation](#-research-motivation)
- [Repository Structure](#️-repository-structure)
- [Experimental Environment](#-experimental-environment)
- [Phase 1: Fruit Classification (ResNet50)](#phase-1-fruit-classification-using-resnet50)
- [Phase 2: Detection + Defect Segmentation (YOLOv8-Seg)](#phase-2-fruit-detection--defect-segmentation-using-yolov8-seg)
- [Phase 3: Semantic Segmentation (U-Net + ResNet50)](#phase-3-semantic-segmentation-using-u-net--resnet50-encoder)
- [Phase 4: Transformer Segmentation (SegFormer)](#phase-4-transformer-based-segmentation-using-segformer)
- [Cross-Model Comparison](#-cross-model-comparison)
- [Setup & Usage](#-setup--usage)
- [Limitations & Future Work](#-limitations--future-work)
- [Learning Outcomes](#-learning-outcomes)
- [Acknowledgments](#-acknowledgments)
- [License](#-license)

---

## 🎯 Project Overview

Fruit quality inspection is traditionally manual, slow, and inconsistent. This project investigates whether modern computer vision architectures — both CNN-based and Transformer-based — can automate defect detection, localization, and severity estimation for fruits. Four models were implemented and evaluated within a single, unified experimental framework:

| Phase | Task | Architecture | Environment | Dataset Source |
|---|---|---|---|---|
| 1 | Binary classification (fresh vs. rotten) | ResNet50 (transfer learning) | Google Colab | Kaggle |
| 2 | Fruit detection + instance defect segmentation | YOLOv8-Seg | Google Colab | Roboflow (YOLOv8 format)|
| 3 | Pixel-level semantic segmentation + damage % estimation | U-Net + ResNet50 encoder | Google Colab | Roboflow (COCO format) |
| 4 | Transformer-based multi-disease segmentation | SegFormer | VS Code + Remote SSH (UST GPU server) | Roboflow (COCO format) |

Each phase builds on the failure modes of the previous one — classification → detection → CNN segmentation → transformer segmentation — enabling a genuine before/after comparison rather than four unrelated experiments.

---

## 🧠 Research Motivation

CNN-based models (ResNet, YOLO, U-Net) offer efficient feature extraction and fast inference, but their limited receptive field constrains long-range context modeling. Transformer-based models (SegFormer) capture global context via self-attention but are computationally heavier. Prior literature rarely compares both families on the *same* fruit-defect task under a *unified* evaluation protocol — this project's core contribution is exactly that side-by-side comparison, across four models, using both quantitative metrics and qualitative visual inspection.

---


## 🗂️ Repository Structure

```
Automated-Fruit-Quality-Inspection-using-Computer-Vision-Models/
├── README.md
├── LICENSE
├── .gitignore
├── requirements.txt
├── Report/
│   └── REPORT_MAULI_JAIN_311410.pdf 
├── Phase-1_ResNet50_Classification/
│   ├── Phase_1_ResNet50_Classification.ipynb
│   ├── phase_1_graphs.png
│   └── phase_1_results.png
├── Phase-2_YOLOv8-Seg_Segmentation/
│   ├── Phase_2_YOLOv8-Seg_Segmentation.ipynb
│   ├── phase_2_graphs.png
│   └── phase_2_sample_result.jpg
├── Phase-3_U-Net+ResNet50_Semantic_Segmentation/
│   ├── Phase_3_UNet+ResNet50_Semantic_Segmentation.ipynb
│   ├── Phase-3 Graphs/
│   └── Phase-3 Sample Outputs/
└── Phase-4_Segformer_Transformer/
    ├── Phase_4_Segformer.ipynb
    ├── Phase-4 Graphs and Matrices/
    └── Phase-4 Sample Outputs/

```

---

## 💻 Experimental Environment

| Component | Details |
|---|---|
| Dev Environment | Google Colab (Phases 1–3); VS Code + Remote SSH (Phase 4) |
| OS | Ubuntu Linux (GPU server, Phase 4) |
| GPU | NVIDIA GPU (Colab runtime + UST GPU server) |
| Language | Python 3.x |
| DL Frameworks | PyTorch, TensorFlow |
| CV Libraries | OpenCV, Torchvision, scikit-image |
| Dataset Sources | Kaggle, Roboflow |
| Visualization | Matplotlib, NumPy, Seaborn |

---

## Phase 1: Fruit Classification using ResNet50

**Goal:** Establish a classification baseline (fresh vs. rotten) and build practical familiarity with deep learning workflows.

**Dataset:** Multi-category fruit image dataset from Kaggle, split into train/val/test, resized and normalized.

**Architecture:** ResNet50 initialized with pretrained ImageNet weights, base layers frozen as a feature extractor, with a custom classification head added on top:
`Input → GAP (2048) → Dropout(0.5) → FC(512) → Dropout(0.5) → Output (Softmax)`

**Training configuration:**
| Parameter | Value |
|---|---|
| Loss function | Categorical Cross-Entropy |
| Optimizer | Adam |
| Learning rate | 5e-4 |
| Batch size | 32 |
| Epochs | 10 |

**Results (Table 4.1):**
| Metric | Value |
|---|---|
| Accuracy | 92.55% |
| Precision | 94 % |
| Recall | 93 % |
| Inference Time | 16.6073 ms |
| FPS | 162.46 |

**Results:**
- Training accuracy improved from **~59% → ~93%** across 10 epochs; validation accuracy tracked closely, indicating reasonable generalization.
- Training/validation loss showed an overall decreasing trend with minor validation fluctuations.

**Limitation → motivation for Phase 2:** Despite high accuracy, the model showed signs of **overfitting** (limited dataset diversity) and, being classification-only, could not localize or quantify *where* a defect was — motivating the move to detection + segmentation.

---

## Phase 2: Fruit Detection & Defect Segmentation using YOLOv8-Seg

**Goal:** Move from image-level classification to instance-level detection and defect segmentation.

**Dataset:** Higher-quality annotated dataset obtained via Roboflow (better annotation quality, greater diversity, more realistic distribution than Phase 1).

**Reproducing this phase:** requires a Roboflow API key (see [Setup & Usage](#-setup--usage)) and access to the linked Roboflow project.

**Architecture:** YOLOv8-Seg — single-stage detector with 3 modules:
1. **Backbone** — hierarchical multi-scale feature extraction
2. **Neck** — feature fusion (FPN-style, P3/P4/P5)
3. **Head** — parallel detection head (bounding boxes + class) and segmentation head (prototype masks)

**Training configuration:**
| Parameter | Value |
|---|---|
| Model | YOLOv8-Seg (pretrained on COCO) |
| Epochs | 10 |
| Optimizer | Adam |
| Batch size | 16 |
| Image size | 640×640 |
| Platform | Google Colab |

**Results (Table 4.2):**
| Metric | Value |
|---|---|
| Pixel Accuracy | 87.53% |
| Precision | 75.69% |
| Recall | 65.46% |
| IoU | 0.0988 |
| Dice Score | 0.1355 |
| Inference Time | 12.89 ms |
| FPS | 77.55 |

**Discussion:** Strong box-level precision/recall, but low IoU/Dice at the pixel-mask level — the model reliably locates defects but doesn't tightly conform to defect *boundaries*, likely due to limited epochs and small defect pixel footprint relative to fruit area.

**Limitation → motivation for Phase 3:** The precision/recall vs. IoU/Dice divergence directly motivated a dedicated pixel-level semantic segmentation approach.

---

## Phase 3: Semantic Segmentation using U-Net + ResNet50 Encoder

**Goal:** Achieve precise pixel-level defect segmentation and enable automated damage-percentage estimation per fruit.

**Dataset:** Same Roboflow dataset as Phase 2, re-exported in **COCO Segmentation** format (binary masks) instead of YOLO format.

**Architecture:** U-Net encoder-decoder with a pretrained ResNet50 encoder, skip connections, and a bottleneck layer — combining ResNet's residual feature extraction with U-Net's fine-detail-preserving decoder.

**Training configuration:**
| Parameter | Value |
|---|---|
| Epochs | 10 |
| Platform | Google Colab |
| Loss | Dice + BCE / IoU-based |

**Results (Table 4.3):**
| Metric | Value |
|---|---|
| Pixel Accuracy | 96.36% |
| Precision | 88.08% |
| Recall | 83.39% |
| IoU | 0.7610 |
| Dice Score | 0.8589 |
| Inference Time | 7.5 ms |
| FPS | 135.8 |

**Qualitative results:** Successfully segmented and estimated damage % for multiple strawberries in a single image (per-instance damage: 69.61%, 53.95%, 31.13%), a rotten apple (82.85% damage), a defective pear (51.48%), and a defective mango (62.51%) — see full report for overlay visualizations and the prediction-confidence heatmap.

**Limitation → motivation for Phase 4:** As a semantic segmentation model, U-Net's performance degrades on **densely clustered/overlapping fruit**. This motivated testing whether a Transformer-based encoder could better capture global context.

---

## Phase 4: Transformer-Based Segmentation using SegFormer

**Goal:** Evaluate whether a Transformer encoder (self-attention, no positional encoding) improves segmentation over the CNN-based U-Net, while extending scope to **multi-disease classification**.

**Dataset:** Roboflow dataset (COCO Segmentation format) with masks for 3 strawberry diseases: **Anthracnose Fruit Rot, Gray Mold, Powdery Mildew** — obtained via UST's internal GPU server.

**Architecture:** `nvidia/segformer-b0-finetuned-ade-512-512`, fine-tuned via Hugging Face `Trainer` API — hierarchical Transformer encoder + lightweight all-MLP decoder, 4 output classes (background + 3 diseases).

**Training configuration:**
| Parameter | Value |
|---|---|
| Base model | SegFormer-B0 (ADE20K pretrained) |
| Epochs | 10 |
| Framework | Hugging Face Transformers + PyTorch |
| Platform | VS Code, Remote SSH → UST GPU server |

**Results (Table 4.4):**
| Metric | Value |
|---|---|
| Pixel Accuracy | 95.92% |
| Precision | 88.02% |
| Recall | 86.35% |
| IoU | 0.7768 |
| Dice Score | 0.8718 |
| Inference Time | 253.34 ms |
| FPS | 3.95 |

**Disease classification performance** (from normalized confusion matrix):
| Disease | Classification Accuracy |
|---|---|
| Anthracnose Fruit Rot | 97% |
| Gray Mold | 100% |
| Powdery Mildew | 67% (confused with Gray Mold in ~31% of cases) |

**Qualitative results:** Correctly classified low-damage samples as "Healthy" (e.g., 4.46% Gray Mold defect area) and high-damage samples as "Highly Defective" (e.g., 40.39% Powdery Mildew defect area); successfully detected **multiple co-occurring diseases** in a single image (Anthracnose + Gray Mold, 37.99% combined defect area).

**Discussion:** SegFormer edges out U-Net on IoU/Dice/Recall and adds multi-disease detection, but at **~34x higher inference latency** (253 ms vs. 7.5 ms) — a real deployment trade-off, not just a leaderboard number.

---

## 📊 Cross-Model Comparison

### Quantitative (segmentation-capable models only — ResNet50 excluded, classification-only)

| Metric | YOLOv8-Seg (P2) | U-Net + ResNet50 (P3) | SegFormer (P4) |
|---|---|---|---|
| Pixel Accuracy | 87.53% | **96.36%** | 95.92% |
| Precision | 75.69% | 88.08% | 88.02% |
| Recall | 65.46% | 83.39% | **86.35%** |
| IoU | 0.0988 | 0.7610 | **0.7768** |
| Dice Score | 0.1355 | 0.8589 | **0.8718** |
| Inference Time | 12.89 ms | **7.5 ms** | 253.34 ms |
| FPS | 77.55 | **135.8** | 3.95 |

### Qualitative (all four models)

| Model | Type | Primary Task | Key Strength | Key Limitation |
|---|---|---|---|---|
| ResNet50 | CNN | Classification | Fast convergence, high accuracy | No localization; overfit on limited data |
| YOLOv8-Seg | CNN | Detection + Instance Seg. | Fastest end-to-end inference (77.55 FPS) | Weak mask boundary precision (low IoU/Dice) |
| U-Net + ResNet50 | CNN | Semantic Segmentation | Best speed/accuracy balance (135.8 FPS) | Struggles with overlapping/clustered fruit |
| SegFormer | Transformer | Semantic Seg. + Disease Classification | Highest IoU/Dice; multi-disease detection | Slowest inference (3.95 FPS); needs more data |

**Bottom line:** U-Net with a ResNet50 encoder offered the best practical balance of accuracy, boundary preservation, and speed for real-time inspection; SegFormer is the stronger choice when segmentation quality and multi-disease diagnosis matter more than latency.

---

## 🚀 Setup & Usage

```bash
git clone https://github.com/<your-username>/fruit-quality-inspection-cv.git
cd fruit-quality-inspection-cv
pip install -r requirements.txt
```

Run any phase notebook top to bottom in Jupyter or Colab:

```bash
jupyter notebook notebooks/phase1_resnet50_classification.ipynb
```

# Phase 1: 
- Dataset : Kaggle dataset [`sriramr/fruits-fresh-and-rotten-for-classification`](https://www.kaggle.com/datasets/sriramr/fruits-fresh-and-rotten-for-classification).
  Requires a Kaggle API token — generate one from your Kaggle account settings,
  download `kaggle.json`, and upload it when prompted in the notebook's setup cell.

- Sample images folder : Phase 1's qualitative testing cells expect a handful of sample fruit images
  (fresh/rotten apple, banana, orange) placed in `SAMPLE_IMAGES_DIR`
  (`/content/sample_images/` by default). Upload your own test images with matching filenames, or update `sample_filenames` in the config cell.


# Phases 2–3:
- Dataset :  Roboflow fruit defect dataset (private project — see note below).
  Requires your own Roboflow API key. Never hardcode it — the notebooks read it via:
  - **Colab (default, as used in these notebooks):** open the 🔑 Secrets tab in the
    left sidebar, add a secret named `ROBOFLOW_API_KEY`, enable notebook access.
    The notebook reads it with `google.colab.userdata.get("ROBOFLOW_API_KEY")`.
  - **Local Jupyter / non-Colab environments:** set it as an environment variable
    instead and swap that line for `os.environ.get("ROBOFLOW_API_KEY")`:

  ```bash
  export ROBOFLOW_API_KEY="your_key_here"
  ```

  > **Note:** The Roboflow project referenced in Phase 2–3 notebooks (`maulis-workspace-l7bbl/fruit_detection-rdzge`) is a private project. To reproduce these phases, either request access, or substitute your own Roboflow-hosted fruit defect dataset and update the workspace/project names accordingly.

---

## 🔭 Limitations & Future Work

- **Dataset diversity:** Several phases were trained primarily on strawberry-heavy data; broader multi-fruit datasets (which we were not able to find while training the models) would improve generalization (noted qualitatively in Phase 3, where the model still segmented non-strawberry fruits reasonably well despite this).
- **Disease confusion:** Powdery Mildew vs. Gray Mold confusion (Phase 4) stems from visual similarity under varying illumination — more diverse disease-stage samples would help.
- **Training budget:** All models were trained for only 10 epochs given Practice School time constraints; longer training with learning-rate scheduling could close some of the remaining gap between YOLOv8-Seg's box metrics and its mask metrics.
- **Deployment:** A hybrid pipeline — fast YOLO/U-Net for real-time screening, SegFormer for flagged/ambiguous cases — would balance speed and accuracy in a production setting.

---

## 📚 Learning Outcomes

- End-to-end deep learning workflow: dataset curation, annotation (YOLO & COCO formats), transfer learning, training, evaluation, and comparative analysis.
- Practical experience with **PyTorch, TensorFlow, Hugging Face Transformers, Ultralytics YOLO, OpenCV**.
- Industry-style GPU workflows: Google Colab for rapid prototyping, and **Remote SSH into a dedicated GPU server**
- Research skills: literature review, experimental design, quantitative + qualitative evaluation, and technical report writing.

---

## 🙏 Acknowledgments

This project was completed under the mentorship of the project mentor at **UST – Infinity Labs, Thiruvananthapuram**, as part of the **Practice School-I program**, Birla Institute of Technology & Science (BITS), Pilani – Hyderabad Campus.

---

## 👤 Author

**Mauli Jain**
M.Sc. Economics & B.E. Computer Science, BITS Pilani, Hyderabad Campus
<br>
Practice School - 1, UST – Infinity Labs, Thiruvananthapuram (May–Jul 2026)

---

## 📄 License

This project's code is licensed under the MIT License — see [LICENSE](LICENSE). Datasets used are subject to their own respective licenses (Kaggle, Roboflow).
