# Vmoge-eeg-alzhimers-
Generating Syntheic EEG data for Alzhimers Patients using VMOGE Latent DIffusion- Graph Neual Network + DPM in complx Fourier Space
# VMoGE Latent Diffusion Pipeline
### Synthetic EEG Generation for Alzheimer's Disease Detection

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Dataset](https://img.shields.io/badge/Dataset-OpenNeuro_ds004504-orange.svg)](https://openneuro.org/datasets/ds004504)

> Master's Thesis — Biomedical Engineering, Amsterdam UMC, 2026  
> Supervisor: Dr. Odysseas Papakyriakou

---

## Overview

This repository contains the implementation of the **VMoGE Latent Diffusion Pipeline** — a hybrid generative framework for synthesising realistic Alzheimer's Disease (AD) EEG data to address the clinical data scarcity problem in EEG-based AD detection.

The pipeline combines:
- A **Variational Mixture of Graph Neural Experts (VMoGE)** encoder operating in complex Cartesian Fourier space
- A **2D U-Net DDPM** decoder for spectral diffusion
- An **EEGNet** downstream classifier evaluated under strict subject-level cross-validation

### Key Results
| Metric | Value |
|--------|-------|
| Fréchet Distance (mean ± std) | 0.152 ± 0.017 |
| Patient-level AUC (Augmented) | 88.9% |
| Specificity improvement | 73.3% → 93.3% |
| False positives reduced | 8 → 2 (out of 29 HC) |
| Dataset | 44 subjects (29 HC, 15 AD) |

---

## Architecture

```
Raw EEG (19ch, 500Hz)
        ↓
   Hanning Window
        ↓
   rfft → F(k) = R(k) + jI(k)
        ↓
   Signed Log Compression
        ↓
   (B, 2, 19, 80) Tensor
        ↓
┌─────────────────────────┐
│     VMoGE Encoder       │
│  ┌─────────────────┐    │
│  │ δ/θ Expert (GCN)│    │
│  │ α  Expert (GCN) │    │
│  │ β  Expert (GCN) │ ←→ Gating Network
│  │ γ  Expert (GCN) │    │
│  └─────────────────┘    │
│   3-layer GCN per expert│
│   10-20 electrode graph │
└──────────┬──────────────┘
           ↓
    z ∈ [-1,1]^128
           ↓
┌──────────────────────────┐
│    DDPM Decoder (U-Net)  │
│    50-step DDIM sampling │
│    Conditioned on z      │
└──────────┬───────────────┘
           ↓
   irfft → Synthetic EEG
           ↓
    Spectral Bias Correction
           ↓
  Synthetic AD Epoch (19ch, 1000 samples)
```

---

## Dataset

**OpenNeuro ds004504** — Miltiadous et al. (2023)
- Full dataset: 88 subjects (29 HC, 36 AD, 23 FTD)
- This work: 44 subjects (29 HC, 15 AD) — FTD excluded
- Sampling rate: 500 Hz
- Channels: 19 (International 10-20 system)
- Epoch length: 2 seconds (1000 samples)
- Preprocessing: Butterworth bandpass 0.5–45 Hz, A1-A2 re-reference, ASR, ICA (provided in dataset derivatives)

Download: https://openneuro.org/datasets/ds004504

---

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/vmoge-eeg-alzheimers.git
cd vmoge-eeg-alzheimers
pip install -r requirements.txt
```

### Requirements
```
torch>=2.0
numpy
scipy
scikit-learn
mne
matplotlib
pandas
```


## Usage

### 1. Prepare data
```bash
python preprocessing/prepare_epochs.py \
    --data_dir /path/to/ds004504/derivatives \
    --output_dir /path/to/processed \
    --sfreq 500 \
    --epoch_len 2.0 \
    --amplitude_threshold 150
```

### 2. Run full pipeline (5-fold)
```bash
sbatch run_vmogeAD.sh
```

### 3. Evaluate classifier (BL vs AUG)
```bash
sbatch run_ablation.sh
```

### 4. Generate figures
```bash
python evaluation/evaluate_classifier.py \
    --results_dir /path/to/pipeline_results \
    --output_dir results/figures
```

---

## Experimental Design

Three conditions compared under 5-fold stratified subject-level cross-validation:

| Condition | Training Data | Purpose |
|-----------|--------------|---------|
| **BL** | Real HC + Real AD | Baseline reference |
| **AUG** | Real HC + Real AD + Synthetic AD | Main experimental condition |
| ~~TSTR~~ | ~~Real HC + Synthetic AD only~~ | ~~Removed from final evaluation~~ |

**Validation:** Subject-level split — no patient's epochs appear in both training and test. Prevents EEG biometric leakage.

---

## Key Design Decisions

| Decision | Choice | Reason |
|----------|--------|--------|
| Spectral representation | Complex Cartesian FFT | Preserves phase by construction |
| Encoder architecture | VMoGE (Mixture of Graph Experts) | Captures electrode spatial topology |
| GCN depth | 3 layers | Covers full fronto-occipital axis without oversmoothing |
| Latent dimension | 128 | Matched to AD training data scale |
| Latent sampling | Empirical (real patient codes + perturbation) | Preserves patient variability vs Gaussian |
| Training strategy | Two-phase (warm-up → joint) | Prevents conditioning collapse |
| Inference | DDIM (50 steps) | 20× faster than DDPM with equivalent quality |

---

## Citation

If you use this work, please cite:

```bibtex
@mastersthesis{elmiari2026vmoge,
  title     = {Synthetic EEG Data Generation for Alzheimer's Disease Detection 
               Using the VMoGE Latent Diffusion Pipeline},
  author    = {Elmiari, Dina},
  school    = {Amsterdam UMC, Biomedical Engineering},
  year      = {2026},
  month     = {July}
}
```

Also cite the dataset:
```bibtex
@article{miltiadous2023dice,
  title   = {DICE-Net: A Novel Convolution-Transformer Architecture for 
             Alzheimer Detection in EEG Signals},
  author  = {Miltiadous, Andreas and others},
  journal = {IEEE Access},
  volume  = {11},
  pages   = {71840--71858},
  year    = {2023}
}
```

---

## Acknowledgements

- Dataset: Miltiadous et al. (2023), AHEPA General Hospital, Thessaloniki
- Compute: Hinton HPC cluster, Tesla V100-PCIE-32GB
- Supervisor: Dr. Odysseas Papakyriakou, Amsterdam UMC

---

## License

MIT License — see [LICENSE](LICENSE) for details.
