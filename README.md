# Recognizing-Images-of-Handwritten-Digits

## 📌 Overview
This project implements a **template-based handwritten digit classifier** using the **MNIST dataset**.  
Instead of training a complex neural network, the system represents each digit (0–9) using a **single template image** created from training samples and classifies test images using distance metrics.

The project evaluates **four model configurations** by combining:
- Mean vs Median templates  
- Euclidean vs Manhattan distance  

This approach is lightweight, interpretable, and computationally efficient, making it a useful **baseline for image classification**.

---

## 📊 Dataset
- **Dataset:** MNIST Handwritten Digits
- **Image size:** 28 × 28 grayscale
- **Training samples:** 60,000
- **Test samples:** 10,000
- **Classes:** Digits 0–9

All images are normalized to the range **[0, 1]** and flattened into **784-dimensional vectors**.

---

## 🧠 Methodology

### Template Construction
For each digit class:
- **Mean Template:** Pixel-wise average of all training images
- **Median Template:** Pixel-wise median of all training images

### Distance Metrics
- **Euclidean Distance (L2)**
- **Manhattan Distance (L1)**

Each test image is assigned the label of the template with the **minimum distance**.

---

## 🔬 Model Configurations
1. Mean + Euclidean  
2. Mean + Manhattan  
3. Median + Euclidean  
4. Median + Manhattan  

---

## 📈 Results Summary

| Model | Overall Accuracy |
|------|------------------|
| **Mean + Euclidean** | **82.03% (Best)** |
| Mean + Manhattan | 66.85% |
| Median + Euclidean | 76.58% |
| Median + Manhattan | 75.35% |

- Digits like **1, 6, and 7** achieved high accuracy
- Digits such as **5, 8, and 9** were more challenging due to handwriting variability
- Euclidean distance consistently outperformed Manhattan distance

---

## 📊 Visualizations
The project generates:
- Mean and Median digit templates
- Per-class accuracy bar plots
- Confusion matrix for the best-performing model

Example outputs:
- `mean_templates.png`
- `median_templates.png`
- `per_class_mean_euclidean.png`
- `cm_mean_euclidean.png`

---

## 🛠 Technologies Used
- **Python**
- **NumPy**
- **Matplotlib**
- Manual MNIST binary file parsing (`idx` format)

---

## 🚀 How to Run

### 1️⃣ Download MNIST Files
Place the following files in the project directory:
- `train-images.idx3-ubyte`
- `train-labels.idx1-ubyte`
- `t10k-images.idx3-ubyte`
- `t10k-labels.idx1-ubyte`

### 2️⃣ Install Dependencies
```bash
pip install numpy matplotlib
