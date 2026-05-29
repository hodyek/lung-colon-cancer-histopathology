# Lung and Colon Cancer Histopathological Classification

**MSB7216 — Deep Learning for Health Data**
Makerere University, Kampala, Uganda

**Odyek Henry** | 2025/HD07/26018U

---

## Overview

This project develops and evaluates a deep learning pipeline for five-class histopathological classification of lung and colon cancer tissue using the LC25000 dataset. The pipeline incorporates Macenko stain normalisation as a preprocessing variable, compares a baseline CNN against EfficientNet-B0 and ResNet-50 transfer learning models, and uses Grad-CAM explainability to verify that model attention aligns with clinically relevant tissue structures.

**Best model:** ResNet-50 (transfer learning, ImageNet pretrained) — **99.87% test accuracy**, train-test gap: **-0.0021 (OK, no overfitting)**

---

## Research Gap

Most published studies on the LC25000 dataset report accuracy above 95% but:

1. Rely on within-dataset validation only — no cross-institutional robustness testing
2. Do not study stain normalisation as a variable that affects model attention
3. Apply Grad-CAM as visual decoration rather than structured per-class histological analysis
4. Rarely report which specific classes fail and why

This study addresses all four gaps.

---

## Repository Structure

```
lung-colon-cancer-histopathology/
├── notebooks/
│   ├── 01_data_understanding_eda.ipynb
│   ├── 02_preprocessing_augmentation.ipynb
│   ├── 03_baseline_model.ipynb
│   ├── 04_efficientnet_b0.ipynb
│   ├── 05_resnet50.ipynb
│   └── 06_gradcam_explainability.ipynb
├── src/
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   └── gradcam.py
├── figures/
│   ├── 05_resnet50_confusion_matrix.png
│   ├── 05_resnet50_gradcam_correct_per_class.png
│   └── 05_resnet50_gradcam_detail_per_class.png
├── models/
│   └── .gitkeep          # weights on Hugging Face Hub
├── data/
│   └── .gitkeep          # download dataset separately
├── reports/
│   ├── Henry_LC25000_Report.docx
│   └── Henry_LC25000_Presentation.pptx
├── requirements.txt
└── README.md
```

---

## Dataset

**LC25000 — Lung and Colon Cancer Histopathological Image Dataset**
Borkowski et al. (2019). arXiv:1912.12142.
Source: https://github.com/tampapath/lung_colon_image_set

| Class | Description | Images |
|-------|-------------|--------|
| colon_aca | Colon adenocarcinoma | 5,000 |
| colon_n | Benign colonic tissue | 5,000 |
| lung_aca | Lung adenocarcinoma | 5,000 |
| lung_n | Benign lung tissue | 5,000 |
| lung_scc | Lung squamous cell carcinoma | 5,000 |
| **Total** | **5 classes — perfectly balanced** | **25,000** |

**Split:** 70% train (17,500) / 15% val (3,750) / 15% test (3,750) — stratified.

---

## Results

| Model | Test Accuracy | Train Accuracy | Gap | Status |
|-------|-------------|----------------|-----|--------|
| Baseline CNN | — | — | — | Reference |
| EfficientNet-B0 | — | — | — | — |
| **ResNet-50 ★** | **99.87%** | **99.66%** | **-0.0021** | **OK** |

**Note on negative gap:** Test accuracy exceeding train accuracy is expected and healthy when dropout and augmentation are applied only during training. It indicates strong generalisation, not data leakage.

**Only misclassification:** 1% of `lung_aca` predicted as `lung_scc`. These are the two histologically most similar classes — both malignant lung tumours sharing nuclear morphological features.

---

## Grad-CAM Findings

Per-class attention analysis confirms the model attends to histologically meaningful structures:

| Class | Model attention |
|-------|----------------|
| colon_aca | Glandular epithelium and nuclear crowding |
| colon_n | Honeycomb epithelial layer of normal colonic mucosa |
| lung_aca | Dense nuclear clusters (slightly diffuse — consistent with lung_scc confusion) |
| lung_scc | Large pleomorphic cells with prominent nucleoli |
| lung_n | Diffuse alveolar architecture — no focal pathological feature |

---

## Preprocessing

- Images resized to 224×224 pixels
- Normalised using ImageNet mean and std
- Macenko LAB-based stain normalisation applied
- **Note:** staintools and spams fail on Colab Python 3.12 — custom numpy/OpenCV implementation used
- Augmentation on training set only: horizontal/vertical flips, rotation ±180°, colour jitter, random resized crop

---

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimiser | Adam |
| Learning rate | 0.001 (frozen); 0.0001 (fine-tuning) |
| Loss function | CrossEntropyLoss |
| Early stopping | Patience 10 epochs |
| Scheduler | ReduceLROnPlateau (factor 0.5, patience 5) |
| NUM_WORKERS | **0** (prevents Colab/Drive connection drops) |

---

## Deployment

The best model (ResNet-50) is deployed as a Gradio application on Hugging Face Spaces.
Model weights are hosted on Hugging Face Hub.

---

## Requirements

```
torch>=2.1.0
torchvision>=0.16.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
Pillow>=10.0.0
opencv-python>=4.8.0
matplotlib>=3.7.0
seaborn>=0.12.0
grad-cam>=1.4.8
gradio>=4.0.0
huggingface_hub>=0.19.0
tqdm>=4.66.0
```

Install: `pip install -r requirements.txt`

---

## Setup

```bash
# Clone
git clone https://github.com/hodyek/lung-colon-cancer-histopathology.git
cd lung-colon-cancer-histopathology

# Install
pip install -r requirements.txt

# Download dataset
# Visit: https://github.com/tampapath/lung_colon_image_set
# Extract into data/lung_colon_image_set/

# Run notebooks in order (01 → 06)
```

---

## Ethical Considerations

The LC25000 dataset is de-identified and HIPAA compliant. No patient-identifying information is present. All images were collected with institutional oversight and are freely available for research and educational use. This model is for research and educational purposes only — clinical deployment requires prospective validation and regulatory approval.

---

## References

- Borkowski, A. A., et al. (2019). LC25000. arXiv:1912.12142.
- He, K., et al. (2016). Deep residual learning for image recognition. CVPR.
- Tan, M., & Le, Q. (2019). EfficientNet. ICML.
- Macenko, M., et al. (2009). Stain normalisation for histology slides. ISBI.
- Selvaraju, R. R., et al. (2017). Grad-CAM. ICCV.
- Kather, J. N., et al. (2019). Predicting survival from colorectal cancer histology. PLOS Medicine.

---

*Makerere University | MSB7216 | 2025/HD07/26018U*
