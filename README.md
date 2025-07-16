# 📄 Paper Summary: *Knowledge Distillation Cross Domain Diffusion Model: A Generative AI Approach for Defect Pattern Segmentation*

[Paper](https://ieeexplore.ieee.org/document/10702557) | [Project Page]() | [Vedio]() | [Code]()

**🙏 Acknowledgement

* This project is supported by the National Science and Technology Council (NSTC), Taiwan, under grant number **NSTC 113-2222-E-A49-006**.

**🎯 Problem & Motivation**

* Semiconductor defect detection often suffers from a lack of pixel-level annotations, making precise segmentation difficult and expensive.

**🔍 Proposed Methods**

1. **ICDDM (Implicit Cross-Domain Diffusion Model)**

   * A weakly supervised image-space diffusion model that learns the joint distribution of defect and clean circuit images.
   * Utilizes denoising score matching through a Markov-chain diffusion process, enabling translation between domains without pixel-level supervision.

2. **CDLDM (Cross-Domain Latent Diffusion Model)**

   * Extends ICDDM by moving diffusion from high-dimensional image space to lower-dimensional latent space via a VAE encoder-decoder.
   * Reduces computational cost while preserving semantic translation between defect and circuit domains.

3. **KDCDDM (Knowledge Distillation Cross-Domain Diffusion Model)**

   * Uses CDLDM as a teacher and trains a GAN-style student to mimic it.
   * Dramatically accelerates inference by distilling the multi-step diffusion into a faster generative model, while maintaining performance.

**⭐ Key Contributions**

* Introduces a **weakly supervised diffusion-based framework** for unsupervised-like defect segmentation.
* Demonstrates **latent-space diffusion** that reduces computation and memory while retaining translation fidelity.
* Proposes a **distillation pathway** (KDCDDM) that achieves high efficiency with minimal loss in quality.

**✅ Impact**

* Provides a robust pipeline for semiconductor defect segmentation and data augmentation, with inference speed suitable for industrial settings.

---

This aligns exactly with your GitHub repo: modules for ICDDM, CDLDM, KDCDDM; VAE and UNet implementations; training and inference scripts; and distillation tools. If you'd like, I can help align benchmarks or extract evaluation results from the paper into your README or training logs.






## Chatbot
Please click [here](https://kdcddm-chatbot.vercel.app/) to enter the chatbot. If you have any questions regarding the KDCDDM paper we've proposed, feel free to ask any questions.

This repository provides a modular framework for semiconductor **defect detection**, **augmentation**, and **cross-domain generation** using diffusion models. It includes support for pixel-space and latent-space modeling, as well as knowledge distillation.

## 🔧 Features

- **ICDDM**: Image-space Cross-Domain Diffusion Model
- **CDLDM**: Cross-Domain Latent Diffusion Model (with VAE)
- **KDCDDM**: Knowledge-Distilled Diffusion with GAN and Student Generator
- **Multi-modal Dataloaders** for wafer map and SEM defect datasets
- **Unified Inference Pipeline** for detection, augmentation, and paired generation
- **Modular Training Scripts** with EMA, cosine scheduler, and loss tracking

---

## 🗂 Directory Structure

```

├── models/               # Diffusion, VAE, UNet, GAN model definitions
├── trainers/             # Training loops for ICDDM, CDLDM, KDCDDM
├── inference/            # Inference tools for detection, augmentation, generation
├── data/                 # Dataset loaders and transforms
├── loss/                 # Diffusion, GAN, KD loss functions
├── utils/                # Logger, EMA, scheduler utilities
├── scripts/              # Shell + export scripts for training and inference
├── configs/              # YAML config files for each model
├── checkpoints/          # Model weight snapshots (output folder)
├── logs/                 # Training logs and metrics
├── samples/              # Generated visual outputs
├── requirements.txt      # Dependencies
└── setup.py              # Installation and CLI entrypoints

````

---

## 🚀 Installation

```bash
# Option 1: Local install
pip install -r requirements.txt
pip install .
````

---

## ⚙️ Training

Run training for each model using the CLI entrypoints:

```bash
train_icddm       # Train ICDDM (pixel-space)
train_cdldm       # Train CDLDM (latent-space)
train_kdcdm       # Train distilled KDCDDM (student + GAN)
```

Each script can be customized via `configs/*.yaml`.

---

## 🔍 Inference

Run inference using scripts under `inference/`:

```bash
# Generate synthetic defects from clean circuit
python inference/defect_detection.py

# Apply latent-space augmentation to defect image
python inference/defect_augmentation.py

# Generate paired (defect, clean) image from input
python inference/paired_generation.py
```

---

## 📦 Export Model

Save a trained model:

```bash
python scripts/export_model.py
```

---

## 🧪 Dataset

Supported inputs:

* **SEM-based Etch Defect Images**
* **Wafer Map Images** (binary or grayscale)
* Define your dataset via `data/etch_sem.py` or `data/wafer_map.py`

---

## 🧰 Configs

Each model is configured via a YAML file in `configs/`:

* `icddm.yaml`
* `cdldm.yaml`
* `kdcdm.yaml`

These define architecture, diffusion steps, learning rate, EMA decay, and loss settings.

---

## 🧠 Citation

Please cite our framework if it helps your research:

```
@misc{defectdiffusion2025,
  title={Defect Diffusion Models for Semiconductor Inspection and Generation},
  author={Your Name},
  year={2025},
  note={https://github.com/yourname/defect_diffusion_models}
}
```

---
