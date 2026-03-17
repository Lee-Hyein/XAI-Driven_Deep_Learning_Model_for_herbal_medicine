# XAI-Driven Deep Learning Model for Herbal Medicine

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> A comparative deep learning framework for classifying herbal medicine images, integrating Explainable AI (XAI) to support sensory evaluation experts in traditional medicine quality assessment.

---

## Overview

This project implements and benchmarks multiple deep learning architectures — **CNN (VGG16)**, **Vision Transformer (ViT)**, and **TransFG** — for herbal medicine image classification. Each model is evaluated with statistical rigor (3 repeated trials, standard deviation, confidence intervals) and paired with XAI visualization methods to provide interpretable predictions.

The goal is to assist herbal medicine sensory evaluation specialists by offering an AI-powered system that is both accurate and explainable.

---

## Model Architectures

### 1. CNN — VGG16
- Traditional Convolutional Neural Network based on the **VGG16** architecture
- Fine-tuned on herbal medicine image dataset via transfer learning
- 3 repeated experiments for statistical reliability
- Grad-CAM visualization for class activation mapping

### 2. ViT — Vision Transformer
- Transformer-based image classification model
- Implemented using the [`timm`](https://github.com/huggingface/pytorch-image-models) library
- Attention map visualization to identify discriminative image regions
- Captures global context through multi-head self-attention

### 3. TransFG — Transformer for Fine-Grained Recognition
- Dual-branch architecture: **Global Branch** + **Local Branch**
- Focuses on fine-grained feature extraction for visually similar herbal species
- Patch importance-based attention analysis
- Designed for high intra-class variance and inter-class similarity

---

## Dataset

| Split       | Description                         |
|-------------|-------------------------------------|
| Train       | Augmented herbal medicine images    |
| Validation  | In-distribution validation samples  |
| Test        | Held-out evaluation set             |

- Herbal medicine image dataset with multi-class labels
- Data augmentation applied (rotation, flipping, color jitter, etc.)
- Structured as standard `ImageFolder` format compatible with PyTorch

> ⚠️ The dataset is not publicly released due to licensing restrictions.  
> Contact the authors for dataset access inquiries.

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install -r requirements.txt

