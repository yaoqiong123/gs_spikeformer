# GS-SpikeFormer: Gated Spiking Attention Transformer for Hyperspectral Image Classification

This repository contains the official PyTorch implementation of the paper:

> **GS-SpikeFormer: Gated Spiking Attention Transformer for Hyperspectral Image Classification**  
> Qiong Yao, Hao Liu, Yingxi Jin, Xiang Xu, Antonio Plaza  
> *Remote Sensing*, 2026

**Paper Link:** [https://doi.org/10.3390/rsxxxxxxxx](https://doi.org/10.3390/rsxxxxxxxx) *(待论文正式发表后替换)*

Citation
If you find this code useful for your research, please cite our paper:
@article{yao2026gsspikeformer,
  title={GS-SpikeFormer: Gated Spiking Attention Transformer for Hyperspectral Image Classification},
  author={Yao, Qiong and Liu, Hao and Jin, Yingxi and Xu, Xiang and Plaza, Antonio},
  journal={Remote Sensing},
  year={2026},
  volume={xx},
  number={x},
  pages={xxx--xxx},
  doi={10.3390/rsxxxxxxxx}
}

## Overview

GS-SpikeFormer is a lightweight spike‑driven Vision Transformer designed for hyperspectral image classification (HSIC). It introduces a **Gated Spiking Attention (GSA)** mechanism that replaces costly matrix multiplications with binary event‑driven feature gating using only logical AND operations. Groupwise convolutions are integrated into both the attention and feed‑forward modules, reducing trainable parameters to just **0.35M** while preserving local spectral‑spatial feature extraction.

### Key Features

- **Gated Spiking Attention (GSA):** Fully spike‑driven, multiplication‑free self‑attention.
- **Lightweight Design:** Only 0.35M parameters, 60‑90% fewer than existing SNN‑based HSIC methods.
- **High Accuracy:** Competitive results on four benchmark datasets with only 4 time steps.
- **Energy Efficient:** Fully binary spike‑driven computation for ultra‑low‑power inference.

## Repository Structure
gs_spikeformer/
├── models/
│ ├── spikingresformer_step2.py # GS‑SpikeFormer model definition
│ └── submodules/
│ ├── layers.py # Custom layers (Conv3x3, Conv1x1, LIF, BN, Linear, SpikingMatmul)
│ └── ...
├── utils/
│ ├── data_preprocess.py # Data loading and preprocessing (PCA, patch extraction)
│ ├── evaluate.py # Evaluation metrics (OA, AA, Kappa, confusion matrix)
│ ├── start.py # Training, testing, and prediction loops
│ ├── auxiliary.py # Helper functions (logging, saving)
│ └── hyper_pytorch.py # Dataset utilities
├── main_IP_spikingresformer.py # Main training script for Indian Pines (IP) dataset
├── requirements.txt # Python dependencies
└── README.md # This file


## Installation

### Prerequisites

- Python 3.8+
- CUDA 11.0+ (recommended for GPU training)
- PyTorch 2.5.1+

### Setup

```bash
# Clone the repository
git clone https://github.com/yaoqiong123/gs_spikeformer.git
cd gs_spikeformer

# Install dependencies
pip install -r requirements.txt

Data Preparation
Download the four benchmark hyperspectral datasets:

Indian Pines (IP) – AVIRIS sensor, 145×145×200, 16 classes

Pavia University (PU) – ROSIS sensor, 610×340×103, 9 classes

Salinas (SA) – AVIRIS sensor, 512×217×204, 16 classes

Houston2013 (HU) – CASI‑1500 sensor, 349×1905×144, 15 classes

Place the dataset files in the ./data/ directory with the following structure:
data/
├── IP/
│   ├── Indian_pines_corrected.mat
│   └── Indian_pines_gt.mat
├── PU/
│   ├── PaviaU.mat
│   └── PaviaU_gt.mat
├── SA/
│   ├── Salinas_corrected.mat
│   └── Salinas_gt.mat
└── HU/
    ├── Houston2013.mat
    └── Houston2013_gt.mat
Note: The data loader automatically applies PCA dimensionality reduction. 
The number of principal components can be adjusted via the components parameter 
in the main script.

Training
To train GS‑SpikeFormer on the Indian Pines dataset:
python main_IP_spikingresformer_pca.py

To train on other datasets, modify the dataset, spatial_size, components, and class_number parameters in the main script:
# Example for Pavia University
dataset = 'PU'
spatial_size = 13
components = 40
class_number = 9

Key Hyperparameters
Parameter	Default	Description
T	4	Number of time steps
heads	8	Number of attention heads
num_groups	16	Number of groups in groupwise convolution
spatial_size	11 (IP), 13 (PU), 17 (SA), 9 (HU)	Input patch size
components	40	Number of PCA components
batch_size	64	Training batch size
epochs	100	Number of training epochs

Evaluation
The model automatically evaluates on the test set after training and reports:

Overall Accuracy (OA)

Average Accuracy (AA)

Kappa coefficient (κ)

Per‑class accuracies

Confusion matrix
