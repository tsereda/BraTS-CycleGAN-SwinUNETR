nice commands

echo "Train images: $(ls processed/brats128_split/train/images/ | wc -l)" && \
echo "Train masks: $(ls processed/brats128_split/train/masks/ | wc -l)" && \
echo "Val images: $(ls processed/brats128_split/val/images/ | wc -l)" && \
echo "Val masks: $(ls processed/brats128_split/val/masks/ | wc -l)" && \
echo "CycleGAN images: $(ls processed/brats128_cyclegan/images/ | wc -l)"


echo "Segmentation images: $(ls ~/processed/brats128_split/segmentation/images/ | wc -l)" && \
echo "Segmentation masks: $(ls ~/processed/brats128_split/segmentation/masks/ | wc -l)" && \
echo "Test images: $(ls ~/processed/brats128_split/test/images/ | wc -l)" && \
echo "Test masks: $(ls ~/processed/brats128_split/test/masks/ | wc -l)" && \
echo "CycleGAN training images: $(ls ~/processed/brats128_split/cyclegan/images/ | wc -l)" && \
echo "Final CycleGAN images: $(ls ~/processed/brats128_cyclegan/images/ | wc -l)"

# BraTS-CycleGAN-SwinUNETR

A comprehensive framework for brain tumor segmentation in MRI images with modality synthesis using 3D CycleGAN and SwinUNETR segmentation models.

## Overview

This repository implements a pipeline for brain tumor segmentation with a focus on handling missing MRI modalities through generative models. It combines two powerful approaches:

1. **3D CycleGAN** - For cross-modality synthesis of missing MRI sequences
2. **SwinUNETR** - For robust 3D brain tumor segmentation

The primary research question addressed is: *Can synthetic MRI modalities generated through CycleGAN effectively replace missing modalities for brain tumor segmentation tasks?*

## Project Structure

```
BraTS-CycleGAN-SwinUNETR/
├── data_preprocessing/           # Preprocessing scripts for BraTS data
├── cyclegan/                     # 3D CycleGAN submodule
├── segmentation/                 # SwinUNETR segmentation model
├── experiments/                  # Experiment scripts
│   ├── train_cyclegan.py         # CycleGAN training script
│   ├── train_segmentation.py     # Segmentation training script 
│   └── evaluate.py               # Evaluation script
├── utils/                        # Utility functions
├── job.yml                       # Kubernetes job for training CycleGAN
├── pod.yml                       # Kubernetes pod for interactive work
├── pvc.yml                       # Kubernetes PVC for data storage
├── environment.yml               # Conda environment specification
├── notebooks/                    # Analysis notebooks
└── README.md                     # This file
```

## Installation

### Prerequisites
- Python 3.9+
- PyTorch 2.0+ (with CUDA support)
- NVIDIA GPU with 16+ GB memory recommended

### Setup Environment

```bash
# Clone repository with submodules
git clone --recursive https://github.com/yourusername/BraTS-CycleGAN-SwinUNETR.git
cd BraTS-CycleGAN-SwinUNETR

# Create conda environment
conda env create -f environment.yml

# Activate environment
conda activate brats-cyclegan
```

## Datasets

This project uses the BraTS 2020 dataset (Brain Tumor Segmentation Challenge) which includes multi-modal MRI scans:
- T1-weighted (T1)
- T1-weighted with contrast enhancement (T1CE)
- T2-weighted (T2)
- T2 Fluid Attenuated Inversion Recovery (FLAIR)

### Data Download (Kaggle)

```bash
# You'll need to set up Kaggle credentials first
kaggle datasets download awsaf49/brats20-dataset-training-validation
unzip brats20-dataset-training-validation.zip -d data/
```

## Data Preprocessing

The preprocessing pipeline converts the raw NIfTI files to a standardized format suitable for deep learning:

```bash
# Preprocess with 50/30/20 split for CycleGAN/Segmentation/Test
python data_preprocessing/preprocess.py --input_train /path/to/BraTS2020_TrainingData \
                                       --input_val /path/to/BraTS2020_ValidationData \
                                       --output_base ./processed_data \
                                       --cyclegan_ratio 0.5 \
                                       --segmentation_ratio 0.3 \
                                       --test_ratio 0.2
```

This creates three main datasets:
- `cyclegan_train`: For training the CycleGAN
- `segmentation_train`: For training the segmentation model
- `final_test`: For final evaluation

## Experiment Workflow

### 1. Train CycleGAN for Modality Synthesis

```bash
# Train T2 -> FLAIR mapping
python experiments/train_cyclegan.py --mapping_type t2_flair \
                                    --data_dir processed_data/experiment_split/cyclegan_train \
                                    --output_base_dir ./output \
                                    --epochs 150 \
                                    --n_resnet 9
```

### 2. Train Segmentation Models

```bash
# Train with complete modalities (baseline)
python experiments/train_segmentation.py --data_path processed_data/experiment_split/segmentation_train \
                                        --output_path output/segmentation/complete \
                                        --modality_mode complete

# Train with modality dropout
python experiments/train_segmentation.py --data_path processed_data/experiment_split/segmentation_train \
                                        --output_path output/segmentation/partial \
                                        --modality_mode partial \
                                        --missing_modality flair

# Train with synthetic modalities
python experiments/train_segmentation.py --data_path processed_data/experiment_split/segmentation_train \
                                        --output_path output/segmentation/augmented \
                                        --modality_mode augmented \
                                        --missing_modality flair \
                                        --cyclegan_path output/t2_flair/checkpoints/best_model.pth
```

### 3. Evaluate Models

```bash
# Evaluate segmentation performance on test set
python experiments/evaluate.py --segmentation_models output/segmentation/*/model_best.pth \
                              --test_path processed_data/experiment_split/final_test \
                              --output_file results/segmentation_results.json
```

## Running on Nautilus NRP

This repository includes Kubernetes manifests for running on the Nautilus Research Platform:

```bash
# Create persistent volume claim
kubectl apply -f pvc.yml

# Run interactive pod
kubectl apply -f pod.yml

# Run CycleGAN training job
kubectl apply -f job.yml
```

## Model Architectures

### 3D CycleGAN
- **Generator**: 3D encoder-decoder with ResNet blocks
- **Discriminator**: 3D PatchGAN discriminator
- **Loss Functions**: Adversarial, Cycle-Consistency, Identity, and Intensity losses

### SwinUNETR Segmentation
- **Encoder**: Hierarchical Swin Transformer blocks
- **Decoder**: Progressive upsampling with feature fusion
- **Loss Function**: Combination of Dice and Cross-Entropy loss

## Monitoring Training

Tensorboard logging is enabled for both CycleGAN and segmentation training:

```bash
tensorboard --logdir=output/
```

## Results Analysis

The repository includes Jupyter notebooks for analyzing results:

- `notebooks/cyclegan_evaluation.ipynb`: Evaluates quality of synthetic modalities
- `notebooks/segmentation_comparison.ipynb`: Compares segmentation performance across different approaches

## Pretraining the Segmentation Model

For better results with limited data, we recommend pretraining the segmentation model:

```bash
# Pretrain on a larger external dataset
python experiments/train_segmentation.py --data_path /path/to/external_dataset \
                                        --output_path output/segmentation/pretrained \
                                        --epochs 50
                                        
# Fine-tune on our segmentation data
python experiments/train_segmentation.py --data_path processed_data/experiment_split/segmentation_train \
                                        --output_path output/segmentation/finetuned \
                                        --pretrained_model output/segmentation/pretrained/model_best.pth \
                                        --epochs 100
```

## Citation

If you use this code in your research, please cite:

```
@misc{brats-cyclegan-swinunetr,
  author = {Your Name},
  title = {BraTS-CycleGAN-SwinUNETR},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/yourusername/BraTS-CycleGAN-SwinUNETR}}
}
```

## References

- [BraTS 2020 Challenge](https://www.med.upenn.edu/cbica/brats2020/)
- [SwinUNETR: Swin Transformers for Semantic Segmentation of Brain Tumors in MRI Images](https://arxiv.org/abs/2201.01266)
- [CycleGAN: Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/abs/1703.10593)

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- BraTS dataset providers
- MONAI Project for medical imaging tools
- Nautilus Research Platform for compute resources