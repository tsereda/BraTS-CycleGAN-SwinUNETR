# BraTS-CycleGAN-SwinUNETR

Advanced brain tumor segmentation using SwinUNETR (Swin Transformer-based U-shaped architecture) combined with CycleGAN for domain adaptation in MRI data.

## Project Overview

This project implements a comprehensive pipeline for brain tumor segmentation using the BraTS 2020 dataset. The pipeline includes:

1. **Data Preprocessing**: Prepares BraTS 2020 data for segmentation training
2. **SwinUNETR Model**: Implementation of a 3D Swin Transformer-based UNet for accurate segmentation
3. **CycleGAN**: Domain adaptation capabilities to improve generalization
4. **Training Pipeline**: Optimized training with mixed precision, gradient accumulation, and advanced loss functions

## Setup in Nautilus

- [USD CS Nautilus Usage Guide](https://docs.google.com/document/d/183yU7vXiyjbO7DGe8N86l4mDyi_KrPGieybGoV7ULcU/edit?tab=t.0)

```bash
# Clone this repository
git clone <your-repo-url>
cd BraTS-CycleGAN-SwinUNETR

#Check for PVCs
kubectl get pvc

# Create PVC if needed 
kubectl apply -f createpvc.yml

#Check for pods
kubectl get pods 

# Create Pod if needed
kubectl apply -f createpod.yml

# Verify pod is running
kubectl logs brats-processing-pod

# Shell into pod
kubectl exec -it brats-processing-pod – bash

# Verify Repo location
ls /opt/repo

# Verify Data location (likely empty)
ls /data 
```

## Dataset

The project uses the BraTS 2020 dataset, which can be downloaded using the provided script:

```bash
todo
```

The dataset includes four MRI modalities:
- T1-weighted (T1)
- T1-weighted with contrast enhancement (T1CE)
- T2-weighted (T2)
- Fluid Attenuated Inversion Recovery (FLAIR)

Each patient case also includes expert segmentation of three tumor regions:
- Enhancing Tumor (ET)
- Tumor Core (TC)
- Whole Tumor (WT)

## Pipeline Steps

### 1. Data Preprocessing

```bash
python data_preprocessing.py
```

The preprocessing pipeline:
- Normalizes each MRI modality
- Crops to a uniform size (128×128×128)
- Creates training/validation splits
- Prepares data for both segmentation and CycleGAN training

### 2. Training

```bash
# Train the segmentation model
python segmentation/train.py
```

The segmentation training includes:
- Mixed precision training for efficiency
- Gradient accumulation for stability
- Advanced loss functions (Dice + Focal loss)
- Learning rate scheduling
- Visualization and metrics tracking

### 3. Validation

```bash
# Validate the preprocessed data
python validate_data.py

# Visualize results (integrated in training)
```

## Model Architecture

### SwinUNETR

The SwinUNETR implementation combines the power of Vision Transformers with the U-Net architecture:

- **Encoder**: Hierarchical Swin Transformer blocks
- **Skip Connections**: Enhanced feature propagation
- **Decoder**: Progressive upsampling with feature fusion

Key components:
- Window-based self-attention with shifted windows
- Patch merging for hierarchical representation
- 3D convolutional decoder blocks

## References

- [SwinUNETR: Swin Transformers for Semantic Segmentation of Brain Tumors in MRI Images](https://arxiv.org/abs/2201.01266)
- [CycleGAN: Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/abs/1703.10593)
- [BraTS 2020 Challenge](https://www.med.upenn.edu/cbica/brats2020/)