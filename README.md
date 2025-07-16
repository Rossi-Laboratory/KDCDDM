# Defect Diffusion Models

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
