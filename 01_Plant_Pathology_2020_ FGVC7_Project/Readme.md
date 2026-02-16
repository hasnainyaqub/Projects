# 🍎 Plant Pathology 2020 – Apple Leaf Disease Classification

![Python](https://img.shields.io/badge/Python-3.12-blue.png)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.png)
![Kaggle](https://img.shields.io/badge/Dataset-Kaggle%20FGVC7-20BEFF?logo=kaggle&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

A deep learning image classification model that detects diseases in apple leaves using transfer learning with **MobileNetV2**, built on the [Plant Pathology 2020 FGVC7](https://www.kaggle.com/c/plant-pathology-2020-fgvc7) Kaggle dataset.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Libraries & Tools](#libraries--tools)
- [Usage](#usage)
- [Conclusion](#conclusion)

---

## Overview

This project builds an automated plant disease diagnosis system using computer vision. Apple leaf images are classified into four disease categories, enabling fast and scalable detection to support agricultural decision-making.

---

## Dataset

The dataset is sourced from the **Plant Pathology 2020 FGVC7** competition on Kaggle. It contains high-resolution RGB images of apple leaves collected under real-world field conditions.

| Class | Description |
|---|---|
| `healthy` | No visible disease |
| `multiple_diseases` | Infected with more than one disease |
| `rust` | Rust fungal infection |
| `scab` | Scab fungal infection |

---

## Methodology

1. **Exploratory Data Analysis** — Loaded and explored the dataset, analyzed class distribution, and visualized sample images.
2. **Preprocessing** — Applied resizing to 224×224, normalization, and data augmentation (flips, rotations, zoom).
3. **Data Splitting** — Split into training and validation sets.
4. **Transfer Learning** — Used MobileNetV2 pretrained on ImageNet as the base model.
5. **Custom Head** — Froze the base model and added dense classification layers.
6. **Training & Tuning** — Trained the model and tuned hyperparameters for optimal performance.
7. **Evaluation** — Assessed model using accuracy, AUC score, and confusion matrix.
8. **Visualization** — Generated predictions and visualized results.

---

## Model Architecture

```
Input (224 × 224 × 3)
        │
        ▼
MobileNetV2 (pretrained on ImageNet, frozen)
        │
        ▼
Global Average Pooling
        │
        ▼
Dense Layer (ReLU)
        │
        ▼
Dense Output Layer (Softmax — 4 classes)
```

| Component | Details |
|---|---|
| Base Model | MobileNetV2 (ImageNet weights) |
| Input Size | 224 × 224 × 3 |
| Base Model | Frozen during training |
| Output | 4-class Softmax |

---

## Results

The model achieves strong classification performance across all four categories. The confusion matrix demonstrates that most predictions fall along the correct diagonal, with minimal misclassifications.

> Misclassifications are analyzed post-training to further improve model robustness and generalization.

---

## Libraries & Tools

| Library | Purpose |
|---|---|
| `TensorFlow / Keras` | Model building and training |
| `NumPy` | Numerical operations |
| `Pandas` | Data loading and manipulation |
| `Matplotlib / Seaborn` | Visualization |
| `Scikit-learn` | Metrics and evaluation |
| `Python 3.12` | Runtime environment |

---

## Usage

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/plant-pathology-2020.git
cd plant-pathology-2020
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Download the Dataset

Download the dataset from [Kaggle](https://www.kaggle.com/c/plant-pathology-2020-fgvc7) and place it in the `data/` directory.

### 4. Run the Notebook

```bash
jupyter notebook plant_pathology_classification.ipynb
```

### 5. Train & Evaluate

Follow the steps inside the notebook to preprocess data, train the model, and evaluate performance.

---

## Conclusion

This project demonstrates how **transfer learning with MobileNetV2** can effectively classify apple leaf diseases from field images. The modular preprocessing and training pipeline is readily adaptable to other agricultural computer vision challenges, making it a strong foundation for real-world plant disease detection systems.